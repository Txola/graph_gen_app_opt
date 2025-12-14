import os
import random
import time

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
    compute_mae=True,
    ensure_validity=False,
):
    cond_values = torch.linspace(
        condition_interval[0], condition_interval[1], batch_size
    ).tolist()

    maes, valids = [], []

    for cond in cond_values:
        mae, validity = run_experiment(
            cfg=cfg,
            sample_steps=num_steps,
            batch_size=1,  # early exit needs batch=1
            condition_value=cond,
            early_exit=True,
            early_exit_start_step=early_exit_start_step,
            compute_mae=compute_mae,
            ensure_validity=ensure_validity,
        )

        if mae is not None:
            maes.append(mae)
        if validity is not None:
            valids.append(validity)

    mean_mae = float(np.mean(maes)) if compute_mae else None
    mean_valid = float(np.mean(valids)) if valids else None

    return mean_mae, mean_valid


def run_steps_experiment(
    sample_steps_list,
    cfg,
    batch_size,
    condition,
    num_folds,
    output_path,
    early_exit,
    compute_mae=True,
    ensure_validity=False,
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

            start_time = time.time()

            if not early_exit:
                mae_no_exit, val_no_exit = run_experiment(
                    cfg,
                    steps,
                    batch_size,
                    condition,
                    early_exit=False,
                    compute_mae=compute_mae,
                    ensure_validity=ensure_validity,
                )

                duration = time.time() - start_time

                all_results.append(
                    {
                        "fold": fold,
                        "steps": steps,
                        "mae_no_exit": mae_no_exit,
                        "validity_no_exit": val_no_exit,
                        "execution_time_sec": duration / batch_size,
                    }
                )

            else:
                mae_exit, val_exit = run_early_exit_batch(
                    cfg=cfg,
                    num_steps=steps,
                    batch_size=batch_size,
                    condition_interval=condition,
                    early_exit_start_step=None,
                    compute_mae=compute_mae,
                    ensure_validity=ensure_validity,
                )

                duration = time.time() - start_time

                all_results.append(
                    {
                        "fold": fold,
                        "steps": steps,
                        "mae_early_exit": mae_exit,
                        "validity_early_exit": val_exit,
                        "execution_time_sec": duration / batch_size,
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
    compute_mae=True,
    ensure_validity=False,
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

            start_time = time.time()

            mae_exit, val_exit = run_early_exit_batch(
                cfg=cfg,
                num_steps=num_steps,
                batch_size=batch_size,
                condition_interval=condition,
                early_exit_start_step=start_step,
                compute_mae=compute_mae,
                ensure_validity=ensure_validity,
            )

            duration = time.time() - start_time

            all_results.append(
                {
                    "fold": fold,
                    "early_exit_start_step": start_step,
                    "mae_early_exit": mae_exit,
                    "validity_early_exit": val_exit,
                    "execution_time_sec": duration / batch_size,
                }
            )

            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)

    print(f"[SAVED] {output_path}")
    return df


def run_repeated_sampling_experiment(
    cfg,
    sample_steps: int,
    repeats: int,
    condition,
    output_path,
    compute_mae=True,
    ensure_validity=False,
):
    """
    Repeatedly sample with batch_size=1 for timing and stability analysis.
    If condition is a tuple (low, high), a random value is drawn each run.
    """

    all_results = []

    interval_mode = isinstance(condition, (tuple, list)) and len(condition) == 2

    for i in range(repeats):
        print(f"  Run {i + 1}/{repeats} (steps={sample_steps})...")
        if interval_mode:
            cond_val = random.uniform(condition[0], condition[1])
        else:
            cond_val = condition

        start_time = time.time()

        mae, validity = run_experiment(
            cfg=cfg,
            sample_steps=sample_steps,
            batch_size=1,
            condition_value=cond_val,
            early_exit=cfg.experiment.early_exit,
            early_exit_start_step=None,
            compute_mae=compute_mae,
            ensure_validity=ensure_validity,
        )

        duration = time.time() - start_time

        all_results.append(
            {
                "steps": sample_steps,
                "run_index": i,
                "condition_value": cond_val,
                "execution_time_sec": duration,
                "mae": mae,
                "validity": validity,
            }
        )

        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)

    return df
