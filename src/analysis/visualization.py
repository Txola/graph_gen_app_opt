import os
import re
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


def plot_histogram(data, title, xlabel, bins=30):
    """Simple histogram."""
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mae_curves(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    mae_col="mae",
    title="MAE vs Steps",
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
    plt.show()


def plot_validity_curves(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    val_col="validity",
    title="Validity vs Steps",
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
    plt.show()


def plot_time_curves(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    time_col="execution_time_sec",
    title="Execution Time vs Steps",
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
    plt.show()


def plot_early_exit_steps(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    mae_col="mae",
    title="MAE with Early Exit vs Steps",
):
    """Plot MAE vs steps for early-exit experiments."""
    plot_mae_curves(csv_files, labels, steps_col, mae_col, title)


def plot_early_exit_starts(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    start_col="early_exit_start_step",
    mae_col="mae",
    title="MAE vs Early Exit Start Step",
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
    plt.show()


def plot_early_exit_timing(
    csv_files: List[str],
    labels: Optional[List[str]] = None,
    steps_col="steps",
    time_col="execution_time_sec",
    title="Execution Time with Early Exit vs Steps",
):
    plot_time_curves(csv_files, labels, steps_col, time_col, title)


def plot_time_histogram_for_step(
    csv_file: str,
    step: int,
    steps_col="steps",
    time_col="execution_time_sec",
    bins=30,
):
    """Histogram of timing for repeated sampling at a given step."""

    df = load_csv(csv_file)
    subset = df[df[steps_col] == step][time_col].values

    title = f"Execution Time Distribution (Step={step})"
    plot_histogram(subset, title, xlabel="Execution Time (sec)", bins=bins)


def plots_simulation(
    csv_path="../../outputs_simulation/fcfs.csv",
    output_folder="../../plots/",
):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    if "response_time" not in df.columns:
        df["response_time"] = df["service_end"] - df["arrival_time"]

    # Compute statistics
    mean_waiting_time = df["waiting_time"].mean()
    mean_service_duration = df["service_duration"].mean()
    mean_response_time = df["response_time"].mean()
    total_customers = len(df)

    # server utilization
    total_simulation_time = df["service_end"].max() - df["arrival_time"].min()
    total_service_time = df["service_duration"].sum()
    server_utilization = total_service_time / total_simulation_time

    print(f"\nSimulation Statistics ({base_name}):")
    print(f"  Total Customers Served: {total_customers}")
    print(f"  Mean Waiting Time: {mean_waiting_time:.4f}")
    print(f"  Mean Service Duration: {mean_service_duration:.4f}")
    print(f"  Mean Response Time: {mean_response_time:.4f}")
    print(f"  Server Utilization: {server_utilization:.4f}")

    # Waiting Time
    plt.figure(figsize=(8, 5))
    sns.histplot(df["waiting_time"], bins=20, kde=True, color="skyblue")
    plt.title(f"Waiting Time Distribution ({base_name})")
    plt.xlabel("Waiting Time")
    plt.ylabel("Frequency")
    plt.grid(True)

    out_path = os.path.join(output_folder, f"{base_name}_waiting_time_hist.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")

    # Service Duration
    plt.figure(figsize=(8, 5))

    sns.histplot(df["service_duration"], bins=20, kde=True, color="salmon")
    plt.title(f"Service Duration Distribution ({base_name})")
    plt.xlabel("Service Duration")
    plt.ylabel("Frequency")
    plt.grid(True)

    out_path = os.path.join(output_folder, f"{base_name}_service_duration_hist.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")

    # Response Time
    plt.figure(figsize=(8, 5))
    sns.histplot(df["response_time"], bins=20, kde=True, color="lightgreen")
    plt.title(f"Response Time Distribution ({base_name})")
    plt.xlabel("Response Time")
    plt.ylabel("Frequency")
    plt.grid(True)
    out_path = os.path.join(output_folder, f"{base_name}_response_time_hist.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")

    # # Percentile Curve
    # df_sorted = df.sort_values("waiting_time")
    # df_sorted["p"] = df_sorted.index / len(df_sorted)

    # plt.figure(figsize=(8, 5))
    # plt.plot(df_sorted["p"], df_sorted["waiting_time"])
    # plt.title(f"Waiting Time Percentile Curve ({base_name})")
    # plt.xlabel("Percentile")
    # plt.ylabel("Waiting Time")
    # plt.grid(True)

    # out_path = os.path.join(output_folder, f"{base_name}_waiting_time_percentiles.png")
    # plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # plt.close()
    # print(f"[OK] Saved: {out_path}")


def plot_mae_per_load(
    load_csv_files, output_path="mae_per_load.png", optimized_model=False
):
    """Plot MAE vs Load Levels from given CSV files."""
    plt.figure(figsize=(9, 5))

    load_levels = []
    maes = []

    mae_csv_path = (
        "../../src/csvs/mae_estimates_opt.csv"
        if optimized_model
        else "../../src/csvs/mae_estimates.csv"
    )
    mae_estimator = MAEEstimator(mae_csv_path)

    for load_csv_file in load_csv_files:
        load = float(load_csv_file.split("_")[-1].split(".csv")[0])

        df_load = pd.read_csv(load_csv_file)
        steps = df_load["sample_steps"].unique()

        maes_per_load = [mae_estimator.estimate(step) for step in steps]
        maes.append(np.mean(maes_per_load))
        load_levels.append(load)

    load_levels, maes = zip(*sorted(zip(load_levels, maes)))

    # plot
    plt.plot(load_levels, maes, "-o")
    plt.xlabel("Load Levels")
    plt.ylabel("Estimated MAE")
    plt.title("Estimated MAE vs Load Levels")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_response_time_per_load(
    load_csv_files, output_path="response_time_per_load.png"
):
    """
    Plot Response Time vs Load Levels from given CSV files.
    """
    plt.figure(figsize=(9, 5))

    load_levels = []
    response_times = []

    for load_csv_file in load_csv_files:
        load = float(load_csv_file.split("_")[-1].split(".csv")[0])
        load_levels.append(load)

        df = pd.read_csv(load_csv_file)
        response_time_per_load = df["response_time"].mean()
        response_times.append(response_time_per_load)

    load_levels, response_times = zip(*sorted(zip(load_levels, response_times)))

    plt.plot(load_levels, response_times, "--s")
    plt.xlabel("Load Levels")
    plt.ylabel("Estimated Response Time")
    plt.title("Estimated Response Time vs Load Levels")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
