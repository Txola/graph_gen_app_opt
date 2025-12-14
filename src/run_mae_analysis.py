import os

import hydra
from analysis.mae_analysis import run_early_exit_start_step_experiment
from analysis.mae_analysis import run_repeated_sampling_experiment
from analysis.mae_analysis import run_steps_experiment
from omegaconf import DictConfig
from omegaconf import OmegaConf


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("\n===== MAE Experiment Runner =====")
    print(OmegaConf.to_yaml(cfg))

    out_dir = os.getcwd()
    mode = cfg.experiment.mode

    compute_mae = cfg.experiment.compute_mae
    ensure_validity = cfg.experiment.ensure_validity

    if mode == "steps":
        _ = run_steps_experiment(
            sample_steps_list=cfg.experiment.sample_steps_list,
            cfg=cfg,
            batch_size=cfg.experiment.batch_size,
            condition=tuple(cfg.experiment.condition_interval),
            num_folds=cfg.experiment.num_folds,
            early_exit=cfg.experiment.early_exit,
            output_path=os.path.join(out_dir, cfg.experiment.output_filename),
            compute_mae=compute_mae,
            ensure_validity=ensure_validity,
        )

    elif mode == "early_exit_start_step":
        _ = run_early_exit_start_step_experiment(
            early_exit_start_steps=cfg.experiment.early_exit_start_steps,
            cfg=cfg,
            batch_size=cfg.experiment.batch_size,
            condition=tuple(cfg.experiment.condition_interval),
            num_steps=cfg.experiment.num_steps,
            num_folds=cfg.experiment.num_folds,
            output_path=os.path.join(out_dir, cfg.experiment.output_filename),
            compute_mae=compute_mae,
            ensure_validity=ensure_validity,
        )

    elif mode == "time_compare":
        bs = 1
        _ = run_steps_experiment(
            sample_steps_list=cfg.experiment.sample_steps_list,
            cfg=cfg,
            batch_size=bs,
            condition=tuple(cfg.experiment.condition_interval),
            num_folds=cfg.experiment.num_folds,
            early_exit=cfg.experiment.early_exit,
            output_path=os.path.join(out_dir, cfg.experiment.output_filename),
            compute_mae=False,
            ensure_validity=True,
        )

    elif mode == "repeated_time":
        _ = run_repeated_sampling_experiment(
            cfg=cfg,
            sample_steps=cfg.experiment.num_steps,
            repeats=cfg.experiment.repeats,
            condition=tuple(cfg.experiment.condition_interval),
            output_path=os.path.join(out_dir, cfg.experiment.output_filename),
            compute_mae=cfg.experiment.compute_mae,
            ensure_validity=cfg.experiment.ensure_validity,
        )

    else:
        raise ValueError(f"Unknown experiment mode: {mode}")


if __name__ == "__main__":
    main()
