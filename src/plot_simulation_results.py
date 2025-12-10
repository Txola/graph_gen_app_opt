import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from analysis.visualization import plots_simulation, plot_mae_per_load, plot_response_time_per_load
import sys


if __name__ == "__main__":
    args = sys.argv
    exp_name = args[1]
    
    optimized_model = False
    
    if exp_name == "fcfs_base":
        folder = "../outputs_simulation/base_model_70_steps_FCFS/"
        files = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")
        ]
    elif exp_name == "rr_base":
        folder = "../outputs_simulation/base_model_70_steps_RR/"
        files = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")
        ]
    elif exp_name == "rr_opt":
        folder = "../outputs_simulation/optimized_model_70_steps_RR/"
        files = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")
        ]
        optimized_model = True
    elif exp_name == "rr_opt_dynamic":
        folder = "../outputs_simulation/optimized_model_dynamic_steps_RR/"
        files = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")
        ]
        optimized_model = True
    else:
        print("Usage:")
        print("  python plot_simulation_results.py fcfs_base")
        print("  python plot_simulation_results.py rr_base")
        print("  python plot_simulation_results.py rr_opt")
        print("  python plot_simulation_results.py rr_opt_dynamic")
        sys.exit(1)
    
    output_folder = Path("../plots/") / exp_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        plots_simulation(csv_path=file, output_folder=output_folder)
        print(f"Plots saved for {file} in {output_folder}")
    plot_mae_per_load(load_csv_files=files, optimized_model=optimized_model, output_path=output_folder / "mae_per_load.png")
    print(f"MAE plot saved for {files[0]} in {output_folder}")
    plot_response_time_per_load(load_csv_files=files, output_path=output_folder / "response_time_per_load.png")
    print(f"Response time plot saved for {files[0]} in {output_folder}")