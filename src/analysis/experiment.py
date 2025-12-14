import torch
from flow_matching.sampler import load_transformer_model
from flow_matching.sampler import QM9CondSampler
from lookup_table import LookupTable
from metrics.molecular_metrics import Evaluator
from metrics.qm9_info import QM9Infos
from models.extra_features import ExtraFeatures
from models.extra_features import ExtraMolecularFeatures
from omegaconf import DictConfig


def build_sampler(cfg: DictConfig, sample_steps: int, params: dict = None):
    """Builds model + sampler, optionally using hyperparameter lookup table."""

    qm9_infos = QM9Infos()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_transformer_model(cfg, qm9_infos, device)

    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=qm9_infos,
    )

    domain_features = ExtraMolecularFeatures(dataset_infos=qm9_infos)
    evaluator = Evaluator()

    if params is not None:
        eta = params.get("eta")
        omega = params.get("omega")
        distortion = params.get("distortion")
    elif cfg.experiment.use_lookup:
        lookup = LookupTable()
        params = lookup.get_best_params(sample_steps)

        eta = params.get("eta")
        omega = params.get("omega")
        distortion = params.get("distortion")
    else:
        eta = 0
        omega = 0.05
        distortion = "polydec"

    sampler = QM9CondSampler(
        cfg,
        qm9_dataset_infos=qm9_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        model=model,
        evaluator=evaluator,
        eta=eta,
        omega=omega,
        distortion=distortion,
    )

    return sampler, evaluator


def run_experiment(
    cfg,
    sample_steps: int,
    batch_size: int,
    condition_value,
    early_exit: bool,
    early_exit_start_step: int = None,
    compute_mae: bool = True,
    ensure_validity: bool = False,
    params: dict = None,
):
    """Run a sampling experiment with optional evaluation and validity enforcement."""

    sampler, evaluator = build_sampler(cfg, sample_steps, params)

    while True:
        samples, labels = sampler.sample(
            batch_size=batch_size,
            sample_steps=sample_steps,
            condition_value=condition_value,
            early_exit=early_exit,
            early_exit_start_step=early_exit_start_step,
        )

        validity = evaluator.compute_validity(samples)

        if ensure_validity and validity != 1.0:
            continue
        break

    if compute_mae:
        mae = evaluator.cond_sample_metric(samples, labels, num_eval=batch_size)
    else:
        mae = None

    return mae, validity
