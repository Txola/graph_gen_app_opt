import numpy as np
import pandas as pd
import sys
from simulation.simulation import Server
from omegaconf import OmegaConf

OUTPUT_DIR = "/home/group-2/asier_graph_gen_app_opt/graph_gen_app_opt/outputs_simulation/"


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


def main():
    cfg = OmegaConf.load("../configs/config.yaml")

    cli_args = parse_cli_arguments()

    n_jobs = cli_args.get("n_jobs", 10)
    mean_inter_arrival = cli_args.get("mean_inter_arrival", 0.1)
    sample_steps = cli_args.get("sample_steps", 100)
    condition_value = cli_args.get("condition_value", -400)

    if "schedule" in cli_args:
        cfg.sample.schedule = cli_args["schedule"]

    if "quantum" in cli_args:
        cfg.sample.quantum = cli_args["quantum"]

    if "retry_invalid_graphs" in cli_args:
        cfg.sample.retry_invalid_graphs = bool(cli_args["retry_invalid_graphs"])

    rng = np.random.default_rng(seed=42)
    inter_arrival_times = rng.exponential(scale=mean_inter_arrival, size=int(n_jobs))

    df_jobs = pd.DataFrame({
        "inter_arrival_time": inter_arrival_times,
        "sample_steps": np.full(int(n_jobs), sample_steps),
        "condition_value": np.full(int(n_jobs), condition_value)
    })

    output_name = cli_args.get("output", "simulation_output.csv")
    output_path = OUTPUT_DIR + output_name

    server = Server(df_jobs, cfg, n_workers=1)
    results = server.run(save=True, output_name=output_path)

    print("Simulation finished!")
    print(results.head())


if __name__ == "__main__":
    main()
