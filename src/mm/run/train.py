import hydra

from .experiment import Experiment, ExperimentConfig


@hydra.main(config_path="../configs")
def main(cfg: ExperimentConfig) -> None:
    Experiment.run_experiment(cfg, lambda e: e.fit())


if __name__ == "__main__":
    main()
