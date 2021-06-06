from typing import Sequence

import hydra
from hydra.utils import instantiate

from .configs import *
from .experiment import Experiment
from .resolvers import *


@hydra.main(config_name="train", config_path="../configs")
def main(cfg: ExperimentConfig) -> Sequence[float]:
    e: Experiment = instantiate(cfg)
    return e.fit()


if __name__ == "__main__":
    customize_args()
    main()
