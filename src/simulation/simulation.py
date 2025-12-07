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

        # Sampling init
        self.qm9_infos = QM9Infos()
        self.extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=self.qm9_infos
        )
        self.domain_features = ExtraMolecularFeatures(dataset_infos=self.qm9_infos)
        self.cfg = cfg
        self.model = load_transformer_model(self.cfg, self.qm9_infos, "cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = Evaluator()


    def _receive(self):
        job_index, job = next(self.job_it)

        # Wait the arrival time
        time.sleep(job.inter_arrival_time)
        arrival_time = time.time() - self.start_time

        if self.cfg.sample.schedule == "RR":
            sampler = QM9CondSampler(
                self.cfg,
                qm9_dataset_infos=self.qm9_infos,
                extra_features=self.extra_features,
                domain_features=self.domain_features,
                model=self.model,
                evaluator=self.evaluator,
                eta=0, omega=1,
                distortion="polydec"
            )
            sampler.initialize(
                batch_size=1,
                sample_steps=int(job.sample_steps),
                condition_value=job.condition_value
            )
        else:
            sampler = None

        return {
            "arrival_time": arrival_time,
            "sample_steps": job.sample_steps,
            "condition_value": job.condition_value,
            "sampler": sampler,
            "finished": False,
            "service_start": None
        }
    
    def _check_valid_graph(self, sampler):
        n = sampler.n_nodes[0]
        atom_types = sampler.X[0, :n].argmax(dim=-1).cpu().numpy().astype(int)
        edge_types = sampler.E[0, :n, :n].argmax(dim=-1).cpu().numpy().astype(int)
        
        valid = self.evaluator.compute_validity([[atom_types, edge_types]])
        
        return valid == 1.0




    def _work(self, task_descriptor):
        if self.cfg.sample.schedule == "FCFS":
            return self._work_fcfs(task_descriptor)
        elif self.cfg.sample.schedule == "RR":
            return self._work_rr(task_descriptor)


    def _work_fcfs(self, task_descriptor):
        if task_descriptor["service_start"] is None:
            task_descriptor["service_start"] = time.time() - self.start_time

        service_start = task_descriptor["service_start"]

        sampler = QM9CondSampler(
            self.cfg,
            qm9_dataset_infos=self.qm9_infos,
            extra_features=self.extra_features,
            domain_features=self.domain_features,
            model=self.model,
            evaluator=self.evaluator,
            eta=0, omega=1,
            distortion="polydec"
        )

        finished = False
        while not finished:
            samples, _ = sampler.sample(
                batch_size=1,
                sample_steps=int(task_descriptor["sample_steps"]),
                condition_value=task_descriptor["condition_value"]
            )

            if getattr(self.cfg.sample, "rerun_if_invalid", False):
                finished = self._check_valid_graph(sampler)
            else:
                finished = True

        service_end = time.time() - self.start_time

        return {
            "service_start": service_start,
            "service_end": service_end,
            "service_duration": service_end - service_start,
            "finished": True
        }

        
    def _work_rr(self, task_descriptor):
        if task_descriptor["service_start"] is None:
            task_descriptor["service_start"] = time.time() - self.start_time

        service_start = task_descriptor["service_start"]
        sampler = task_descriptor["sampler"]
        quantum = self.cfg.sample.quantum

        for _ in range(quantum):
            if sampler.finished:
                break
            sampler.step()

        if getattr(self.cfg.sample, "rerun_if_invalid", False) and sampler.finished:
            if not self._check_valid_graph(sampler):
                sampler.finished = False

        service_end = time.time() - self.start_time

        return {
            "service_start": service_start,
            "service_end": service_end,
            "service_duration": service_end - service_start,
            "finished": sampler.finished
        }



    def submitter(self):
        queue_ = self.job_queue[0]
        arrivals = np.zeros(self.max_count)

        for i in range(self.max_count):
            task = self._receive()
            queue_.put(task)
            arrivals[i] = task["arrival_time"]

        print("Done submitting")
        self.running = False
        return arrivals,


    def worker(self, idx=0):
        queue_ = self.job_queue[idx]
        results = []

        schedule = self.cfg.sample.schedule

        while self.running or not queue_.empty():
            if not queue_.empty():
                task = queue_.get()
                queue_.task_done()

                work_info = self._work(task)

                finished = work_info.get("finished", True)

                if finished:
                    results.append({
                        "arrival_time": task["arrival_time"],
                        "service_start": work_info["service_start"],
                        "service_end": work_info["service_end"],
                        "service_duration": work_info["service_duration"]
                    })
                    self.progress.update(1)

                else:
                    # print("Return to the queue")
                    task["finished"] = False
                    queue_.put(task)

        print(f"Worker {idx} finished.")
        return results




    def run(self, save=True, output_name="simulation_output.csv"):
        self.running = True
        self.start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_workers = [
                executor.submit(self.worker, i) for i in range(self.n_workers)
            ]
            future_submitter = executor.submit(self.submitter)

            arrivals = future_submitter.result()[0]
            worker_results = [f.result() for f in future_workers]

        results = worker_results[0]

        df = pd.DataFrame(results)

        df["waiting_time"] = df["service_start"] - df["arrival_time"]
        df["system_time"] = df["service_end"] - df["arrival_time"]

        if save:
            df.to_csv(output_name, index=False)
            print(f"Saved results to {output_name}")

        return df
