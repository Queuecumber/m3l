import hydra
from hydra.utils import instantiate

from mm.run.experiment import Experiment, ExperimentConfig


@hydra.main(config_path="../configs")
def main(cfg: ExperimentConfig) -> None:
    experiment: Experiment = instantiate(cfg)
    experiment.fit()


if __name__ == "__main__":
    main()
