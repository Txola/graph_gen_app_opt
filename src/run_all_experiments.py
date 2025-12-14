import subprocess
import sys


def run(cmd):
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)


def main():
    args = sys.argv

    base_cmd = ["python", "run_simulation.py", "n_jobs=1000"]

    if args[1] == "fcfs_opt":
        for load in [1.4, 1.6, 1.8, 2.0, 2.2]:
            # 1. Optimized, 70 steps, FCFS
            cmd = base_cmd + [
                "schedule=FCFS",
                "optimized_model=true",
                "dynamic_steps=false",
                "run_last_jobs=true",
                f"mean_inter_arrival={load}",
            ]
            run(cmd)

    elif args[1] == "rr_opt_dynamic":
        for load in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            # 2. Optimized model, dynamic steps, RR
            cmd = base_cmd + [
                "schedule=RR",
                "optimized_model=true",
                "dynamic_steps=true",
                "run_last_jobs=true",
                f"mean_inter_arrival={load}",
            ]
            run(cmd)
    elif args[1] == "fcfs_opt_dynamic":
        for load in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            # 3. Optimized model, dynamic steps, FCFS
            cmd = base_cmd + [
                "schedule=FCFS",
                "optimized_model=true",
                "dynamic_steps=true",
                "run_last_jobs=true",
                f"mean_inter_arrival={load}",
            ]
            run(cmd)

    elif args[1] == "fixed_load_fcfs":
        for Q_sat in [15, 30, 35, 40, 50, 55, 60, 75]:
            # 4. Fixed load, FCFS
            cmd = base_cmd + [
                "schedule=FCFS",
                "optimized_model=true",
                "dynamic_steps=true",
                "run_last_jobs=true",
                "mean_inter_arrival=1.0",
                f"Q_sat={Q_sat}",
            ]
            run(cmd)

    else:
        print("Usage:")
        print("  python run_all_experiments.py fcfs_opt")
        print("  python run_all_experiments.py rr_opt_dynamic")
        print("  python run_all_experiments.py fcfs_opt_dynamic")
        print("  python run_all_experiments.py fixed_load_fcfs")
        return


if __name__ == "__main__":
    main()
