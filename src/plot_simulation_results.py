import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from analysis.visualization import plots_simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results from a CSV file")
    parser.add_argument("--csv_file", type=str, default="../outputs_simulation/fcfs.csv", help="Path to the CSV file")
    parser.add_argument("--output_folder", type=str, default="../plots/", help="Folder to save plots")
    args = parser.parse_args()

    plots_simulation(args.csv_file, args.output_folder)
