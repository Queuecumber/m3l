from pathlib import Path

import hydra
from mme.models import QGAC
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_resolver("path", lambda x: Path(x))


@hydra.main(config_path="../configs")
def main(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
