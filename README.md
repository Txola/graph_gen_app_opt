# Energy-Conditioned Molecular Generation with DeFoG

## Introduction
This repository contains the code for a molecular graph generation service based on the **DeFoG (Discrete Flow Matching for Graphs)** framework, available at:  
https://github.com/manuelmlmadeira/DeFoG

The goal of the project is to generate **chemically valid molecular graphs conditioned on a target energy value**, while studying the impact of different sampling strategies, hyperparameter configurations, and scheduling policies.

In addition to molecule generation, this repository includes a **simulation framework** that evaluates system-level performance under different workloads by executing the actual generative model. This enables realistic measurements of service time, waiting time, and response time under controlled conditions.

All experiments, simulations, and analyses presented in the accompanying report can be reproduced using the scripts provided here.

---

## Repository Structure
The repository is organized as follows:
```
├── plots/ # Contain all the generated plots
├── src/
├── environment.yaml 
```

### `src/`
This directory contains all executable scripts used during the project to run the experiments, and that are necessary to execute to replicate the results (`sample.py`, `run_simulation.py`, `run_all_experiments.py`, `run_mae_analysis.py`).

---

## Environment Setup

### 1. Install Miniconda
If Conda is not installed, download and install Miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. Create the Conda Environment
Create the environment using the provided configuration file:
```bash
conda env create -f environment.yaml
```

### 3. Activate the Environment
```bash
conda activate defog
```

## Running the Code

1. **Generate a Single Molecule**  
    The `sample.py` script allows you to generate a single molecular graph conditioned on a target energy value and reports its validity.
    Example execution:
    ```
    python sample.py energy=-1.5 steps=70 eta=25 omega=0.5                    ########################################################################################### Cambiar
    ```
    This script:
   - Runs the DeFoG conditional sampler
   - Generates a molecular graph
   - Evaluates and prints its validity

2. **Run a Queueing Simulation**  
    The `run_simulation.py` script executes a full system-level simulation where incoming requests are processed by a molecular generation server.
    Example execution:
    ```
    python run_simulation.py n_jobs=50 mean_inter_arrival=0.1 schedule=RR optimized_model=false dynamic_steps=false retry_invalid_graphs=true run_last_jobs=true
    ```
    This script:
   - Executes the actual generative model for each request
   - Supports FCFS and Round Robin scheduling
   - Records waiting time, service time, response time, and retries
   - Outputs results to a CSV file

## Reproducing the Experiments

1. **Run All Simulation Experiments**  
    To reproduce all timing and scheduling experiments reported in the paper, run the `run_all_experiments.py` script:
    ```
    python run_all_experiments.py
    ```
    This script executes all predefined simulations and stores the resulting metrics.

2. **MAE Analysis**  
    To reproduce the Mean Absolute Error (MAE) analysis for energy-conditioned generation, run the `run_mae_analysis.py` script:
    ```
    python run_mae_analysis.py
    ```
    This script generates molecules across different target energies and computes MAE statistics.



## Visualizing Results

All plots used in the report are generated from the Jupyter notebook `plots.ipynb` located in principal directory. ########################################################################## Cambiar
The notebook loads the CSV outputs from the simulation and analysis scripts and produces the final figures.

## Notes
- The simulations do not approximate service times: each job executes the real DeFoG model and generates an actual molecular graph.
- GPU execution is recommended for reasonable runtimes. And to replicate the exact results be care of the GPU used for the experiments (is mentioned in the report).
- Most of the experiments are fully configurable via command-line arguments and configuration files.






















