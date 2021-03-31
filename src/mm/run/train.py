import hydra

from .experiment import Experiment, ExperimentConfig
from hydra.utils import instantiate


@hydra.main(config_name="train", config_path="../configs")
def main(cfg: ExperimentConfig) -> None:
    e: Experiment = instantiate(cfg)
    e.fit()


if __name__ == "__main__":
    main()
