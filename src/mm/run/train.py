import hydra
from hydra.utils import instantiate

from .experiment import Experiment, ExperimentConfig


@hydra.main(config_name="train", config_path="../configs")
def main(cfg: ExperimentConfig) -> None:
    e: Experiment = instantiate(cfg)
    e.fit()


if __name__ == "__main__":
    main()
