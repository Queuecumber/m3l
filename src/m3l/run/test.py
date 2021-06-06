from typing import Any, Mapping, Sequence

import hydra
from hydra.utils import instantiate

from .configs import *
from .experiment import Experiment
from .resolvers import *


@hydra.main(config_name="test", config_path="../configs")
def main(cfg: ExperimentConfig) -> Sequence[Mapping[str, Any]]:
    e: Experiment = instantiate(cfg)
    return e.test()


if __name__ == "__main__":
    customize_args()
    main()
