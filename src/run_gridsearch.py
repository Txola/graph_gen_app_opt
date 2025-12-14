from itertools import product

import hydra
import pandas as pd
from analysis.experiment import run_experiment
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    df = pd.DataFrame()
    num_steps = [10, 25, 40, 55, 70, 85, 100]
    etas = [10, 25, 50, 75, 150]
    omegas = [0, 0.5, 1]

    for step, eta, omega in product(num_steps, etas, omegas):
        print(f"Running experiment with steps={step}, eta={eta}, omega={omega}")
        _, validity = run_experiment(
            cfg,
            sample_steps=step,
            batch_size=1000,
            condition_value=(-460, -350),
            early_exit=False,
            compute_mae=False,
            ensure_validity=False,
            params={"eta": eta, "omega": omega, "distortion": "polydec"},
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "steps": [step],
                        "eta": [eta],
                        "omega": [omega],
                        "validity": [validity],
                    }
                ),
            ],
            ignore_index=True,
        )
        df.to_csv("gridsearch_results_etas.csv", index=False)


if __name__ == "__main__":
    main()
