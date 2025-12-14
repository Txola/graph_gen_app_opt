import time

import hydra
import torch
from flow_matching.sampler import load_transformer_model
from flow_matching.sampler import QM9CondSampler
from metrics.molecular_metrics import Evaluator
from metrics.qm9_info import QM9Infos
from models.extra_features import ExtraFeatures
from models.extra_features import ExtraMolecularFeatures
from omegaconf import DictConfig
from rdkit import RDLogger


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    RDLogger.DisableLog("rdApp.*")

    qm9_infos = QM9Infos()
    model = load_transformer_model(
        cfg, qm9_infos, "cuda" if torch.cuda.is_available() else "cpu"
    )

    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=qm9_infos,
    )
    domain_features = ExtraMolecularFeatures(dataset_infos=qm9_infos)
    evaluator = Evaluator()
    sampler = QM9CondSampler(
        cfg,
        qm9_dataset_infos=qm9_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        model=model,
        evaluator=evaluator,
        eta=cfg.sample.eta,
        omega=cfg.sample.omega,
        distortion="polydec",
    )
    print("Sampling...")
    start = time.time()
    samples, labels = sampler.sample(
        batch_size=1,
        sample_steps=cfg.sample.sample_steps,
        condition_value=cfg.sample.condition_value,
    )
    end = time.time()
    print(f"Sampling took {end - start:.4f} seconds.")

    start = time.time()
    validity = evaluator.compute_validity(samples)
    end = time.time()
    print(f"Validity computation took {end - start:.4f} seconds.")
    if validity == 1.0:
        print("The graph is valid")
    else:
        print("The graph is not valid")


if __name__ == "__main__":
    main()
