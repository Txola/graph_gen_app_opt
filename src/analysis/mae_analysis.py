import os

import numpy as np
import pandas as pd
import torch
from analysis.experiment import run_experiment


def run_early_exit_batch(
    cfg,
    num_steps,
    batch_size,
    condition_interval,
    early_exit_start_step=None,
):
    """
    Runs sampling `batch_size` times because early exit only works with batch_size=1.
    """

    cond_values = torch.linspace(
        condition_interval[0], condition_interval[1], batch_size
    ).tolist()

    maes, valids = [], []
    cnts = 0

    for cond in cond_values:
        mae, validity, len_valid = run_experiment(
            cfg=cfg,
            sample_steps=num_steps,
            batch_size=1,
            condition_value=cond,
            early_exit=True,
            early_exit_start_step=early_exit_start_step,
        )

        if mae < 0:
            continue

        maes.append(mae)
        valids.append(validity)
        cnts += len_valid

    return float(np.mean(maes)), float(np.mean(valids)), cnts


def run_steps_experiment(
    sample_steps_list, cfg, batch_size, condition, num_folds, output_path, early_exit
):
    """
    Runs the experiment for num_folds folds.
    If early_exit is True, uses early exit sampling.
    """

    if os.path.exists(output_path):
        print(f"Reusing existing global results: {output_path}")
        df = pd.read_csv(output_path)
        first_fold = df["fold"].max() + 1
        all_results = df.to_dict("records")
    else:
        first_fold = 0
        all_results = []

    for fold in range(first_fold, num_folds + first_fold):
        print(f"\n===== FOLD {fold + 1}/{num_folds} =====")

        for steps in sample_steps_list:
            print(f"  Running {steps} steps...")

            if not early_exit:
                mae_no_exit, val_no_exit, len_valids = run_experiment(
                    cfg, steps, batch_size, condition, early_exit=False
                )

                all_results.append(
                    {
                        "fold": fold,
                        "steps": steps,
                        "mae_no_exit": mae_no_exit,
                        "validity_no_exit": val_no_exit,
                        "num_valids_no_exit": len_valids,
                    }
                )
            else:
                mae_exit, val_exit, cnt_exit = run_early_exit_batch(
                    cfg=cfg,
                    num_steps=steps,
                    batch_size=batch_size,
                    condition_interval=condition,
                    early_exit_start_step=None,
                )

                all_results.append(
                    {
                        "fold": fold,
                        "steps": steps,
                        "mae_early_exit": mae_exit,
                        "validity_early_exit": val_exit,
                        "num_valids_early_exit": cnt_exit,
                    }
                )

            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)
    print(f"[SAVED] {output_path}")
    return df


def run_early_exit_start_step_experiment(
    early_exit_start_steps,
    cfg,
    batch_size,
    condition,
    num_steps,
    num_folds,
    output_path,
):
    """
    Runs an experiment varying early_exit_start_step with fixed num_steps.
    """
    if os.path.exists(output_path):
        print(f"Reusing existing results: {output_path}")
        return pd.read_csv(output_path)

    all_results = []

    for fold in range(num_folds):
        print(f"\n===== FOLD {fold + 1}/{num_folds} =====")

        for start_step in early_exit_start_steps:
            print(f"  Running early_exit_start_step={start_step}...")

            mae_exit, val_exit, cnt_exit = run_early_exit_batch(
                cfg=cfg,
                num_steps=num_steps,
                batch_size=batch_size,
                condition_interval=condition,
                early_exit=True,
                early_exit_start_step=start_step,
            )
            all_results.append(
                {
                    "fold": fold,
                    "early_exit_start_step": start_step,
                    "mae_early_exit": mae_exit,
                    "validity_early_exit": val_exit,
                    "num_valids_early_exit": cnt_exit,
                }
            )

            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)
    print(f"[SAVED] {output_path}")

    return df
