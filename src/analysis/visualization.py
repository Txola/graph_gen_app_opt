import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_tensor(x):
    """Convert 'tensor(14.2853)' → 14.2853."""
    if isinstance(x, str) and x.startswith("tensor"):
        return float(re.findall(r"tensor\((.*)\)", x)[0])
    return float(x)


def bootstrap_ci(values, n_samples=2000, low=5, high=95):
    """Obtain bootstrap confidence intervals for the mean of the given values."""
    values = np.array(values)
    N = len(values)

    boot_means = []
    for _ in range(n_samples):
        sample = np.random.choice(values, size=N, replace=True)
        boot_means.append(sample.mean())

    return np.percentile(boot_means, low), np.percentile(boot_means, high)


def load_stats(csv_file, steps_col, mae_col, validity_col, count_col):
    df = pd.read_csv(csv_file)
    df[mae_col] = df[mae_col].apply(clean_tensor)
    df = df[df[count_col] > 0].copy()

    grouped = df.groupby(steps_col)

    steps, mae_mean, mae_low, mae_high = [], [], [], []
    val_mean, val_low, val_high = [], [], []

    for step, g in grouped:
        steps.append(step)

        maes = g[mae_col].values
        valids = g[validity_col].values

        mae_mean.append(maes.mean())
        val_mean.append(valids.mean())

        lo_m, hi_m = bootstrap_ci(maes)
        lo_v, hi_v = bootstrap_ci(valids)

        mae_low.append(lo_m)
        mae_high.append(hi_m)
        val_low.append(lo_v)
        val_high.append(hi_v)

    return (
        np.array(steps),
        np.array(mae_mean),
        np.array(mae_low),
        np.array(mae_high),
        np.array(val_mean),
        np.array(val_low),
        np.array(val_high),
    )


def plot_mae(
    csv_files,
    labels=None,
    steps_col="steps",
    mae_col="mae_no_exit",
    validity_col="validity_no_exit",
    count_col="num_valids_no_exit",
    title="MAE (Energy) vs Steps",
):
    """Plot MAE vs Steps from given CSV files."""

    if labels is None:
        labels = csv_files

    plt.figure(figsize=(9, 5))

    for csv_file, label in zip(csv_files, labels):
        label = label.split(".csv")[0].split("/")[-1]

        steps, mae_mean, mae_low, mae_high, _, _, _ = load_stats(
            csv_file, steps_col, mae_col, validity_col, count_col
        )

        plt.plot(steps, mae_mean, "-o", label=label)
        plt.fill_between(steps, mae_low, mae_high, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("MAE (Energy)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_validity(
    csv_files,
    labels=None,
    steps_col="steps",
    mae_col="mae_no_exit",
    validity_col="validity_no_exit",
    count_col="num_valids_no_exit",
    title="Validity vs Steps",
):
    """Plot Validity vs Steps from given CSV files."""

    if labels is None:
        labels = csv_files

    plt.figure(figsize=(9, 5))

    for csv_file, label in zip(csv_files, labels):
        label = label.split(".csv")[0].split("/")[-1]

        steps, _, _, _, val_mean, val_low, val_high = load_stats(
            csv_file, steps_col, mae_col, validity_col, count_col
        )

        plt.plot(steps, val_mean, "--s", label=label)
        plt.fill_between(steps, val_low, val_high, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("Validity")
    plt.ylim(0.25, 1)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plots_simulation(
    csv_path="../../outputs_simulation/fcfs.csv", output_folder="../../plots/"
):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Waiting Time
    plt.figure(figsize=(8, 5))
    plt.hist(df["waiting_time"], bins=40, edgecolor="black")
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
    plt.hist(df["service_duration"], bins=40, edgecolor="black")
    plt.title(f"Service Duration Distribution ({base_name})")
    plt.xlabel("Service Duration")
    plt.ylabel("Frequency")
    plt.grid(True)

    out_path = os.path.join(output_folder, f"{base_name}_service_duration_hist.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")

    # System Time
    plt.figure(figsize=(8, 5))
    plt.hist(df["system_time"], bins=40, edgecolor="black")
    plt.title(f"System Time Distribution ({base_name})")
    plt.xlabel("System Time")
    plt.ylabel("Frequency")
    plt.grid(True)
    out_path = os.path.join(output_folder, f"{base_name}_system_time_hist.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")

    # Percentile Curve
    df_sorted = df.sort_values("waiting_time")
    df_sorted["p"] = df_sorted.index / len(df_sorted)

    plt.figure(figsize=(8, 5))
    plt.plot(df_sorted["p"], df_sorted["waiting_time"])
    plt.title(f"Waiting Time Percentile Curve ({base_name})")
    plt.xlabel("Percentile")
    plt.ylabel("Waiting Time")
    plt.grid(True)

    out_path = os.path.join(output_folder, f"{base_name}_waiting_time_percentiles.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")
