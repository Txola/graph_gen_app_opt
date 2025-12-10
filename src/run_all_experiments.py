import subprocess
import sys
import numpy as np

def run(cmd):
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

def main():
    
    args = sys.argv
    loads = np.arange(1.0, 4.1, 0.5).tolist()


    # common parameters
    base_cmd = ["python", "run_simulation.py", "n_jobs=500"]
    
    for load in loads:
        if args[1] == "fcfs_base":
            # 1. Base model, 70 steps, FCFS
            cmd = base_cmd + [
                "schedule=FCFS",
                "optimized_model=false",
                "dynamic_steps=false",
                "run_last_jobs=true",
                f"mean_inter_arrival={load}"
            ]
            run(cmd)

        elif args[1] == "rr_base":
            # 2. Base model, 70 steps, RR
            cmd = base_cmd + [
                "schedule=RR",
                "optimized_model=false",
                "dynamic_steps=false",
                "run_last_jobs=true",
                f"mean_inter_arrival={load}"
            ]
            run(cmd)

        elif args[1] == "rr_opt":
            # 3. Optimized model, static 70 steps, RR
            cmd = base_cmd + [
                "schedule=RR",
                "optimized_model=true",
                "dynamic_steps=false",
                # maybe false?
                "run_last_jobs=true",
                f"mean_inter_arrival={load}"
            ]
            run(cmd)

        elif args[1] == "rr_opt_dynamic":
            # 4. Optimized model, dynamic steps, RR
            cmd = base_cmd + [
                "schedule=RR",
                "optimized_model=true",
                "dynamic_steps=true",
                # maybe false?
                "run_last_jobs=true",
                f"mean_inter_arrival={load}"
            ]
            run(cmd)

        else:
            print("Usage:")
            print("  python run_all_experiments.py fcfs_base")
            print("  python run_all_experiments.py rr_base")
            print("  python run_all_experiments.py rr_opt")
            print("  python run_all_experiments.py rr_opt_dynamic")
            return


if __name__ == "__main__":
    main()
