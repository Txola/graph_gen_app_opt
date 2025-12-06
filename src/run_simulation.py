import numpy as np
import pandas as pd
from simulation import Server

def main():
    n_jobs = 10

    mean_inter_arrival = 0.1 
    rng = np.random.default_rng(seed=42)
    inter_arrival_times = rng.exponential(scale=mean_inter_arrival, size=n_jobs)

    sample_steps = np.full(n_jobs, 100)
    condition_values = np.full(n_jobs, -400)

    df_jobs = pd.DataFrame({
        "inter_arrival_time": inter_arrival_times,
        "sample_steps": sample_steps,
        "condition_value": condition_values
    })

    from omegaconf import OmegaConf
    cfg = OmegaConf.load("../configs/config.yaml")
    
    server = Server(df_jobs, cfg, n_workers=1)
    results = server.run(save=True, output_name="/home/group-2/asier_graph_gen_app_opt/graph_gen_app_opt/outputs_simulation/rr.csv")

    print("Simulation finished!")
    print(results.head())

if __name__ == "__main__":
    main()
