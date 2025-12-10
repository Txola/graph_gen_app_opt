import numpy as np
import pandas as pd
import sys
from simulation.simulation import Server
from omegaconf import OmegaConf
from pathlib import Path


def parse_cli_arguments():
    args_dict = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)

            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                pass

            args_dict[key] = value
    return args_dict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("yes", "true", "t", "1")
    return bool(v)


def main():
    cfg = OmegaConf.load("../configs/config.yaml")

    cli_args = parse_cli_arguments()

    n_jobs = cli_args.get("n_jobs", 10)
    mean_inter_arrival = cli_args.get("mean_inter_arrival", 0.1)
    sample_steps = cli_args.get("sample_steps", 70)

    if "schedule" in cli_args:
        cfg.sample.schedule = cli_args["schedule"]

    if "quantum" in cli_args:
        cfg.sample.quantum = cli_args["quantum"]

    if "retry_invalid_graphs" in cli_args:
        cfg.sample.retry_invalid_graphs = str2bool(cli_args["retry_invalid_graphs"])
        
    if "dynamic_steps" in cli_args:
        cfg.sample.dynamic_steps = str2bool(cli_args["dynamic_steps"])
        
    if "optimized_model" in cli_args:
        cfg.sample.optimized_model = str2bool(cli_args["optimized_model"])
        print(f"Optimized model set to: {cfg.sample.optimized_model}")
    
    if "run_last_jobs" in cli_args:
        cfg.sample.run_last_jobs = str2bool(cli_args["run_last_jobs"])
        print(f"Run last jobs set to: {cfg.sample.run_last_jobs}")
    
    print(cfg.sample)
    rng = np.random.default_rng(seed=42)
    inter_arrival_times = rng.exponential(scale=mean_inter_arrival, size=int(n_jobs))

    df_jobs = pd.DataFrame({
        "inter_arrival_time": inter_arrival_times,
        "sample_steps": np.full(int(n_jobs), sample_steps),
        "condition_value": np.random.uniform(-460, -350, size=int(n_jobs))
    })
    repo_root = Path(__file__).resolve().parents[1]
    if not cfg.sample.optimized_model and cfg.sample.run_last_jobs and not cfg.sample.dynamic_steps and cfg.sample.schedule == "FCFS":
        exp_name = f"base_model_70_steps_FCFS"
    elif not cfg.sample.optimized_model and cfg.sample.run_last_jobs and not cfg.sample.dynamic_steps and cfg.sample.schedule == "RR":
        exp_name = f"base_model_70_steps_RR"
    elif cfg.sample.optimized_model and cfg.sample.run_last_jobs and not cfg.sample.dynamic_steps and cfg.sample.schedule == "RR":
        exp_name = f"optimized_model_70_steps_RR"
    elif cfg.sample.optimized_model and cfg.sample.run_last_jobs and cfg.sample.dynamic_steps and cfg.sample.schedule == "RR":
        exp_name = f"optimized_model_dynamic_steps_RR"
    else:
        raise ValueError("Configuration combination not supported for naming convention.")
    OUTPUT_DIR = repo_root / "outputs_simulation" / exp_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"arrivals_{mean_inter_arrival}"
    output_path = OUTPUT_DIR / output_name

    server = Server(df_jobs, cfg, n_workers=1)
    print("Starting simulation...")
    results = server.run(save=True, output_name=output_path)

    print("Simulation finished!")
    print(results.head())


if __name__ == "__main__":
    main()
