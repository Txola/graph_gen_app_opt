import time

import hydra
from flow_matching.sampler import QM9CondSampler
from metrics.molecular_metrics import compute_validity
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

    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=qm9_infos,
    )
    domain_features = ExtraMolecularFeatures(dataset_infos=qm9_infos)
    sampler = QM9CondSampler(
        cfg,
        qm9_dataset_infos=qm9_infos,
        extra_features=extra_features,
        domain_features=domain_features,
        eta=0,
        omega=1,
        distortion="polydec",
    )
    print("Sampling...")
    start = time.time()
    samples, _ = sampler.sample(
        batch_size=1,
        sample_steps=100,
        condition_value=-400,
    )
    end = time.time()
    print(f"Sampling took {end - start:.4f} seconds.")

    start = time.time()
    print(compute_validity(samples))
    end = time.time()
    print(f"Validity computation took {end - start:.4f} seconds.")


if __name__ == "__main__":
    main()
