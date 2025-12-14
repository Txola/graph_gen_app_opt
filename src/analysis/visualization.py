import os
import re
from pathlib import Path
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from metrics.mae_estimator import MAEEstimator


def clean_tensor(x):
    """Convert 'tensor(14.2853)' → 14.2853."""
    if isinstance(x, str) and x.startswith("tensor"):
        return float(re.findall(r"tensor\((.*)\)", x)[0])
    return float(x)


def load_csv(csv_path):
    """Load CSV and return DataFrame."""

    df = pd.read_csv(csv_path)
    for col in df.columns:
        df[col] = df[col].apply(clean_tensor)
    return df


def load_per_load_csvs(folder: str, suffix=".csv"):
    """Return sorted list of CSV files in folder."""
    folder_path = Path(folder)
    files = [f for f in folder_path.glob(f"*{suffix}")]
    files = sorted(files, key=lambda f: float(f.stem.split("_")[-1]))
    return files


def bootstrap_ci(values, n_samples=2000, low=5, high=95):
    """Obtain bootstrap confidence intervals for the mean of the given values."""
    values = np.array(values)
    N = len(values)
    boot_means = []
    for _ in range(n_samples):
        sample = np.random.choice(values, size=N, replace=True)
        boot_means.append(sample.mean())
    return np.percentile(boot_means, low), np.percentile(boot_means, high)


def compute_stats(df: pd.DataFrame, value_col: str, group_col: str):
    """
    Compute mean and bootstrap confidence intervals for error bars.
    Returns (x, mean, ci_low, ci_high)
    """
    x = []
    mean = []
    ci_low = []
    ci_high = []

    for step, group in df.groupby(group_col):
        vals = group[value_col].values
        x.append(step)
        mean_val = np.mean(vals)
        mean.append(mean_val)
        low, high = bootstrap_ci(vals)
        ci_low.append(low)
        ci_high.append(high)

    return np.array(x), np.array(mean), np.array(ci_low), np.array(ci_high)


def plot_curve_with_band(x, mean, low, high, label):
    """Line with shaded confidence / variability band."""
    plt.plot(x, mean, "-o", label=label)
    plt.fill_between(x, low, high, alpha=0.2)


def plot_histogram(
    data, title, xlabel, bins=60, hist_color="skyblue", kde_color="darkred"
):
    plt.figure(figsize=(8, 4))

    _, bin_edges, _ = plt.hist(
        data, bins=bins, alpha=1, density=True, color=hist_color, label="Histogram"
    )

    xmin, xmax = bin_edges[0], bin_edges[-1]

    sns.kdeplot(data, clip=(xmin, xmax), color=kde_color, lw=2, label="KDE", alpha=0.4)
    plt.xlim(xmin, xmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_mae_curves(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    mae_col="mae",
    title="MAE vs Steps",
    save_path="mae_comp.pdf",
):
    """MAE vs Steps for multiple CSV files."""

    if labels is None:
        labels = [f.split("/")[-1].replace(".csv", "") for f in csv_files]

    plt.figure(figsize=(9, 5))

    for csv_path, label in zip(csv_files, labels):
        df = load_csv(csv_path)
        x, mean, low, high = compute_stats(df, mae_col, steps_col)
        plot_curve_with_band(x, mean, low, high, label)

    plt.xlabel("Steps")
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_validity_curves(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    val_col="validity",
    title="Validity vs Steps",
    save_path="validity_comp.pdf",
):
    """Validity vs Steps."""

    if labels is None:
        labels = [f.split("/")[-1].replace(".csv", "") for f in csv_files]

    plt.figure(figsize=(9, 5))

    for csv_path, label in zip(csv_files, labels):
        df = load_csv(csv_path)
        x, mean, low, high = compute_stats(df, val_col, steps_col)
        plot_curve_with_band(x, mean, low, high, label)

    plt.xlabel("Steps")
    plt.ylabel("Validity")
    plt.ylim(0, 1)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_time_curves(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    time_col="execution_time_sec",
    title="Execution Time vs Steps",
    save_path="time_comp.pdf",
):
    """Execution time vs Steps."""

    if labels is None:
        labels = [f.split("/")[-1].replace(".csv", "") for f in csv_files]

    plt.figure(figsize=(9, 5))

    for csv_path, label in zip(csv_files, labels):
        df = load_csv(csv_path)
        x, mean, low, high = compute_stats(df, time_col, steps_col)
        plot_curve_with_band(x, mean, low, high, label)

    plt.xlabel("Steps")
    plt.ylabel("Time (sec)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_early_exit_steps(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    mae_col="mae",
    title="MAE with Early Exit vs Steps",
    save_path="early_steps.pdf",
):
    """Plot MAE vs steps for early-exit experiments."""
    plot_mae_curves(csv_files, labels, steps_col, mae_col, title)
    plt.savefig(save_path)


def plot_early_exit_starts(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    start_col="early_exit_start_step",
    mae_col="mae",
    title="MAE vs Early Exit Start Step",
    save_path="early_starts.pdf",
):
    """MAE vs where early exit begins."""

    if labels is None:
        labels = [f.split("/")[-1].replace(".csv", "") for f in csv_files]

    plt.figure(figsize=(9, 5))

    for csv_path, label in zip(csv_files, labels):
        df = load_csv(csv_path)
        x, mean, low, high = compute_stats(df, mae_col, start_col)
        plot_curve_with_band(x, mean, low, high, label)

    plt.xlabel("Early Exit Start Step")
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_time_histogram_for_step(
    csv_file: str,
    step: int,
    steps_col="steps",
    time_col="execution_time_sec",
    bins=30,
    save_path="times.pdf",
):
    """Histogram of timing for repeated sampling at a given step."""

    df = load_csv(csv_file)
    subset = df[df[steps_col] == step][time_col].values

    title = f"Execution Time Distribution (Step={step})"
    plot_histogram(subset, title, xlabel="Execution Time (sec)", bins=bins)
    plt.savefig(save_path)
    plt.show()


def plot_tradeoff_per_load(
    load_csv_files, mae_csv_path, output_path="tradeoff_per_load.png", use_lambda=True
):
    """
    Plot Estimated MAE and Average Response Time vs Load (or λ) on the same plot.
    Consistent style with visualization.py.
    """
    load_levels = []
    maes = []
    response_times = []

    mae_estimator = MAEEstimator(mae_csv_path)

    for load_csv_file in load_csv_files:
        load_val = float(
            os.path.basename(load_csv_file).split("_")[-1].split(".csv")[0]
        )
        x_val = 1 / load_val if use_lambda else load_val
        load_levels.append(x_val)

        df = pd.read_csv(load_csv_file)
        steps = df["sample_steps"].unique()
        maes_per_load = [mae_estimator.estimate(step) for step in steps]
        maes.append(np.mean(maes_per_load))

        response_times.append(df["response_time"].mean())

    load_levels, maes, response_times = zip(
        *sorted(zip(load_levels, maes, response_times))
    )

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(load_levels, maes, "-o", color="tab:blue", label="Estimated MAE")
    ax1.set_xlabel("Load (λ)" if use_lambda else "Load (1/λ)")
    ax1.set_ylabel("Estimated MAE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        load_levels, response_times, "-o", color="tab:orange", label="Response Time"
    )
    ax2.set_ylabel("Average Response Time (sec)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Estimated MAE and Response Time vs Load")
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_tradeoff_per_Qsat(
    load_csv_files, mae_csv_path, output_path="tradeoff_per_Qsat.png"
):
    """
    Plot Estimated MAE and Average Response Time vs Q_sat parameter for fixed load (λ=1).
    Consistent style with visualization.py.
    """
    Q_sats = []
    maes = []
    response_times = []

    mae_estimator = MAEEstimator(mae_csv_path)

    for load_csv_file in load_csv_files:
        Q_sat = float(os.path.basename(load_csv_file).split("_")[-1].split(".csv")[0])
        Q_sats.append(Q_sat)

        df = pd.read_csv(load_csv_file)
        steps = df["sample_steps"].unique()
        maes_per_load = [mae_estimator.estimate(step) for step in steps]
        maes.append(np.mean(maes_per_load))

        response_times.append(df["response_time"].mean())

    Q_sats, maes, response_times = zip(*sorted(zip(Q_sats, maes, response_times)))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(Q_sats, maes, "-o", color="tab:blue", label="Estimated MAE")
    ax1.set_xlabel("Q_sat")
    ax1.set_ylabel("Estimated MAE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(14.5, 15.5)

    ax2 = ax1.twinx()
    ax2.plot(Q_sats, response_times, "-o", color="tab:orange", label="Response Time")
    ax2.set_ylabel("Average Response Time (sec)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Estimated MAE and Response Time vs Q_sat (λ=1)")
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_queue_and_steps_per_Qsat(
    load_csv_file,
    output_path="queue_steps_per_Qsat.png",
    queue_col="queue_length",
    steps_col="sample_steps",
):
    """
    Plot queue length and assigned sampling steps per job
    for a fixed Q_sat value.

    Style consistent with visualization.py tradeoff plots.
    """

    Q_sat = float(os.path.basename(load_csv_file).split("_")[-1].split(".csv")[0])

    df = pd.read_csv(load_csv_file)

    job_idx = np.arange(len(df))
    queue_lengths = df[queue_col].values
    assigned_steps = df[steps_col].values

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        job_idx,
        queue_lengths,
        color="tab:blue",
        markersize=3,
        label="Queue Length",
    )
    ax1.set_xlabel("Job Index")
    ax1.set_ylabel("Queue Length", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        job_idx,
        assigned_steps,
        color="tab:orange",
        markersize=3,
        label="Assigned Steps",
    )
    ax2.set_ylabel("Assigned Sampling Steps", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )
    ax1.set_xlim(0, 800)
    ax2.set_xlim(0, 800)

    plt.title(f"Queue Length and Assigned Steps per Job (Q_sat={Q_sat})")
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
