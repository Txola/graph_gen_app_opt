import concurrent.futures
import pandas as pd
import numpy as np
import queue
import time
from tqdm import tqdm
import torch
torch.cuda.empty_cache()

from models.extra_features import ExtraFeatures, ExtraMolecularFeatures
from metrics.qm9_info import QM9Infos
from flow_matching.sampler import QM9CondSampler
from flow_matching.sampler import load_transformer_model
from metrics.molecular_metrics import Evaluator
from lookup_table import LookupTable


class Server:
    def __init__(self, df, cfg, n_workers=1):
        # Server init
        self.job_it = df.iterrows()
        self.max_count = df.shape[0]
        self.job_queue = [queue.Queue() for _ in range(n_workers)]
        self.start_time = 0.
        self.running = False
        self.n_workers = n_workers
        self.progress = tqdm(total=self.max_count, desc="Processing jobs")

        # Sampling parameters (use defaults if not in cfg)
        self.S_min = getattr(cfg.sample, "S_min", 20)
        self.S_max = getattr(cfg.sample, "S_max", 70)
        self.Q_free = getattr(cfg.sample, "Q_free", 0)
        self.Q_sat = getattr(cfg.sample, "Q_sat", 45)
        self.dynamic_steps = getattr(cfg.sample, "dynamic_steps", False)
        self.max_retries = getattr(cfg.sample, "max_retries", 5)
        self.optimized_model = getattr(cfg.sample, "optimized_model", False)
        self.retry_invalid_graphs = getattr(cfg.sample, "retry_invalid_graphs", False)
        self.run_last_jobs = getattr(cfg.sample, "run_last_jobs", False)
        
        if self.optimized_model:
            self.lookup = LookupTable()

        # Sampling init
        self.qm9_infos = QM9Infos()
        self.extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=self.qm9_infos
        )
        self.domain_features = ExtraMolecularFeatures(dataset_infos=self.qm9_infos)
        self.cfg = cfg
        # load model once (cpu or cuda)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_transformer_model(self.cfg, self.qm9_infos, device)
        self.evaluator = Evaluator()

        print("Server initialized.")

    def compute_dynamic_steps(self, Q):
        if Q <= self.Q_free:
            return self.S_max
        if Q >= self.Q_sat:
            return self.S_min

        ratio = (Q - self.Q_free) / (self.Q_sat - self.Q_free)
        S = self.S_max - ratio * (self.S_max - self.S_min)
        return int(S)

    def _receive(self):
        job_index, job = next(self.job_it)

        # Wait the arrival time (simulate inter-arrival)
        time.sleep(job.inter_arrival_time)
        arrival_time = time.time() - self.start_time

        # return minimal task descriptor; sampler/params will be added in worker
        return {
            "arrival_time": arrival_time,
            "sample_steps": int(job.sample_steps),
            "condition_value": job.condition_value,
            "sampler": None,
            "finished": False,
            "service_start": None,
            "retry_count": 0
        }

    def _check_valid_graph(self, samples):
        valid = self.evaluator.compute_validity(samples)
        return valid

    def _work(self, task_descriptor):
        if self.cfg.sample.schedule == "FCFS":
            return self._work_fcfs(task_descriptor)
        elif self.cfg.sample.schedule == "RR":
            return self._work_rr(task_descriptor)
        else:
            raise ValueError(f"Unknown schedule {self.cfg.sample.schedule}")

    def _work_fcfs(self, task_descriptor):
        if task_descriptor["service_start"] is None:
            task_descriptor["service_start"] = time.time() - self.start_time

        service_start = task_descriptor["service_start"]

        # Use safe .get with defaults to avoid KeyError
        eta = task_descriptor.get("eta", 0)
        omega = task_descriptor.get("omega", 0.05)
        distortion = task_descriptor.get("distortion", "polydec")

        sampler = QM9CondSampler(
            self.cfg,
            qm9_dataset_infos=self.qm9_infos,
            extra_features=self.extra_features,
            domain_features=self.domain_features,
            model=self.model,
            evaluator=self.evaluator,
            eta=eta, omega=omega,
            distortion=distortion
        )

        finished = False
        retry_count = -1
        while not finished:
            retry_count += 1
            samples, _ = sampler.sample(
                batch_size=1,
                sample_steps=int(task_descriptor["sample_steps"]),
                condition_value=task_descriptor["condition_value"]
            )

            # print(self.retry_invalid_graphs)
            if self.retry_invalid_graphs:
                finished = self._check_valid_graph(samples)
            else:
                finished = True

            # safety: avoid infinite retries in pathological cases
            if retry_count >= self.max_retries:
                finished = True
                break

        service_end = time.time() - self.start_time

        return {
            "service_start": service_start,
            "service_end": service_end,
            "service_duration": service_end - service_start,
            "finished": True,
            "retry_count": retry_count,
            "sample_steps": task_descriptor["sample_steps"],
        }

    def _work_rr(self, task_descriptor):
        # RR expects a sampler already initialized in the task_descriptor
        if task_descriptor["service_start"] is None:
            task_descriptor["service_start"] = time.time() - self.start_time

        service_start = task_descriptor["service_start"]
        sampler = task_descriptor["sampler"]
        quantum = int(getattr(self.cfg.sample, "quantum", 1))

        retry_count = task_descriptor.get("retry_count", 0)

        # perform quantum steps
        for _ in range(quantum):
            # defensive: if no sampler (shouldn't happen) break
            if sampler is None:
                break
            if sampler.finished:
                break
            sampler.step()

        # if finished, check validity if requested
        if sampler is not None and sampler.finished:
            if self.retry_invalid_graphs:
                if not self._check_valid_graph(sampler):
                    # mark unfinished so worker will requeue (subject to max_retries)
                    retry_count += 1
                    task_descriptor["retry_count"] = retry_count
                    sampler.finished = False

        service_end = time.time() - self.start_time

        return {
            "service_start": service_start,
            "service_end": service_end,
            "service_duration": service_end - service_start,
            "finished": bool(sampler.finished) if sampler is not None else True,
            "retry_count": retry_count,
            "sample_steps": task_descriptor["sample_steps"],
        }

    # Sentinel-based submitter (puts sentinel per worker when done)
    def submitter(self):
        queue_ = self.job_queue[0]
        arrivals = np.zeros(self.max_count)

        for i in range(self.max_count):
            task = self._receive()
            if i == self.max_count - 1:
                task["last_job"] = True  # mark last job
                self.last_job_enqueued = not self.run_last_jobs  # inform workers
            queue_.put(task)
            arrivals[i] = task["arrival_time"]

        # enqueue one sentinel per worker so workers know when to stop
        for _ in range(self.n_workers):
            queue_.put({"__stop__": True})

        print("Done submitting")
        return arrivals

    def worker(self, idx=0):
        queue_ = self.job_queue[idx]
        results = []

        while True:
            task = queue_.get()  # blocking get
            
            # if sentinel -> mark task done and break
            if isinstance(task, dict) and (task.get("__stop__") or task.get("last_job", False) and not self.run_last_jobs):
                results.append({
                    "arrival_time": task.get("arrival_time"),
                    "service_start": None,
                    "service_end": None,
                    "service_duration": None,
                    "sample_steps": task.get("sample_steps"),
                    "condition_value": task.get("condition_value"),
                    "num_in_queue": task.get("num_in_queue_at_start", None),
                    "retry_count": task.get("retry_count", 0),
                    "eta": task.get("eta", None),
                    "omega": task.get("omega", None),
                    "distortion": task.get("distortion", None),
                    "error": "not_executed"
                })
                queue_.task_done()
                break
            

            # if last job enqueued, mark as not executed and continue
            if getattr(self, "last_job_enqueued", False):
                results.append({
                    "arrival_time": task.get("arrival_time"),
                    "service_start": None,
                    "service_end": None,
                    "service_duration": None,
                    "sample_steps": task.get("sample_steps"),
                    "condition_value": task.get("condition_value"),
                    "num_in_queue": task.get("num_in_queue_at_start", None),
                    "retry_count": task.get("retry_count", 0),
                    "eta": task.get("eta", None),
                    "omega": task.get("omega", None),
                    "distortion": task.get("distortion", None),
                    "error": "not_executed"
                })
                queue_.task_done()
                continue

            # if sampler/params not initialized for this task, create them here (thread-safe per task)
            if task.get("sampler") is None:
                # compute dynamic sample_steps based on queue length (optional)
                if self.dynamic_steps:
                    current_queue_length = sum(q.qsize() for q in self.job_queue)
                    sample_steps = self.compute_dynamic_steps(current_queue_length)
                    #  Get sampling parameters from lookup table of mlp predictions
                    if self.optimized_model:
                        params = self.lookup.get_best_params(sample_steps)
                        eta = params.get("eta")
                        omega = params.get("omega")
                        distortion = params.get("distortion")
                    else:   
                        eta, omega, distortion = 0, 0.05, "polydec"
                else:
                    sample_steps = int(task.get("sample_steps", self.S_max))
                    if self.optimized_model:
                        params = self.lookup.get_best_params(sample_steps)
                        eta = params.get("eta")
                        omega = params.get("omega")
                        distortion = params.get("distortion")
                    else:
                        eta, omega, distortion = 0, 0.05, "polydec"

                task["sample_steps"] = int(sample_steps)
                task["eta"] = eta
                task["omega"] = omega
                task["distortion"] = distortion
                task["num_in_queue_at_start"] = queue_.qsize()

                # create sampler only for RR; FCFS builds sampler inside _work_fcfs
                if self.cfg.sample.schedule == "RR":
                    sampler = QM9CondSampler(
                        self.cfg,
                        qm9_dataset_infos=self.qm9_infos,
                        extra_features=self.extra_features,
                        domain_features=self.domain_features,
                        model=self.model,
                        evaluator=self.evaluator,
                        eta=eta, omega=omega,
                        distortion=distortion
                    )
                    sampler.initialize(
                        batch_size=1,
                        sample_steps=int(task["sample_steps"]),
                        condition_value=task["condition_value"]
                    )
                else:
                    sampler = None
                task["sampler"] = sampler

            # run the work (catch exceptions per task so whole simulation doesn't die)
            try:
                work_info = self._work(task)
            except Exception as e:
                # record the error and move on
                results.append({
                    "arrival_time": task.get("arrival_time"),
                    "service_start": task.get("service_start"),
                    "service_end": None,
                    "service_duration": None,
                    "sample_steps": task.get("sample_steps"),
                    "condition_value": task.get("condition_value"),
                    "num_in_queue": task.get("num_in_queue_at_start"),
                    "retry_count": task.get("retry_count", 0),
                    "error": str(e)
                })
                queue_.task_done()
                continue

            finished = work_info.get("finished", True)
            retry_count = work_info.get("retry_count", task.get("retry_count", 0))

            if finished:
                # success: append result and mark done
                results.append({
                    "arrival_time": task["arrival_time"],
                    "service_start": work_info["service_start"],
                    "service_end": work_info["service_end"],
                    "service_duration": work_info["service_duration"],
                    "sample_steps": task.get("sample_steps"),
                    "condition_value": task.get("condition_value"),
                    "num_in_queue": task.get("num_in_queue_at_start"),
                    "retry_count": retry_count,
                    "eta": task.get("eta"),
                    "omega": task.get("omega"),
                    "distortion": task.get("distortion"),
                })
                self.progress.update(1)
                queue_.task_done()
            else:
                # Not finished (RR partial or invalid graph). Requeue if under max_retries.
                task["retry_count"] = retry_count
                if task["retry_count"] < self.max_retries:
                    queue_.put(task)
                else:
                    # give up after max_retries, record as failed
                    results.append({
                        "arrival_time": task.get("arrival_time"),
                        "service_start": task.get("service_start"),
                        "service_end": None,
                        "service_duration": None,
                        "sample_steps": task.get("sample_steps"),
                        "condition_value": task.get("condition_value"),
                        "num_in_queue": task.get("num_in_queue_at_start"),
                        "retry_count": task.get("retry_count"),
                        "error": "max_retries_exceeded",
                        "eta": task.get("eta"),
                        "omega": task.get("omega"),
                        "distortion": task.get("distortion"),
                    })
                queue_.task_done()

        print(f"Worker {idx} finished.")
        return results

    def run(self, save=True, output_name="simulation_output.csv"):
        self.running = True
        self.start_time = time.time()

        print("Simulation started.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # start producer
            future_submitter = executor.submit(self.submitter)
            # start consumers
            future_workers = [executor.submit(self.worker, i) for i in range(self.n_workers)]

            # wait for submitter to finish (it enqueues sentinels)
            arrivals = future_submitter.result()[0]
            print("Submitter finished; sentinels enqueued.")
            print(f"Arrivals: {arrivals}")

            # wait for workers to finish consuming
            worker_results = [f.result() for f in future_workers]

            print("Workers finished; results collected.")

        # combine results from workers
        if self.n_workers == 1:
            results = worker_results[0]
        else:
            results = []
            for w in worker_results:
                results.extend(w)

        df = pd.DataFrame(results)

        if not df.empty:
            df["waiting_time"] = df["service_start"] - df["arrival_time"]
            df["response_time"] = df["service_end"] - df["arrival_time"]
            df["sample_steps"] = [r.get("sample_steps") for r in results]
            df["condition_value"] = [r.get("condition_value") for r in results]
            df["num_in_queue"] = [r.get("num_in_queue") for r in results]
            df["retry_count"] = [r.get("retry_count", 0) for r in results]
            df["eta"] = [r.get("eta") for r in results]
            df["omega"] = [r.get("omega") for r in results]
            df["distortion"] = [r.get("distortion") for r in results]
        

        if save:
            df.to_csv(output_name, index=False)
            print(f"Saved results to {output_name}")

        return df
