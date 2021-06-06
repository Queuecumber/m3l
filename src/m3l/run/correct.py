import hydra
from hydra.utils import instantiate

from .configs import *
from .experiment import Experiment
from .resolvers import *


@hydra.main(config_name="correct", config_path="../configs")
def main(cfg: ExperimentConfig) -> None:
    e: Experiment = instantiate(cfg)
    return e.correct()


if __name__ == "__main__":
    main()
