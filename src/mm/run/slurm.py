import time
from typing import Callable, Generator
import sys

from hydra.core.singleton import Singleton
from omegaconf import DictConfig
from yaspin import yaspin
from submitit import JobEnvironment, Job, SlurmExecutor


class SlurmSubmission:
    def __init__(self, callable: Callable[[DictConfig, JobEnvironment], None]) -> None:
        self.callable = callable

    def __call__(self, cfg: DictConfig, hydra_state: Singleton) -> None:
        Singleton.set_state(hydra_state)
        job_env = JobEnvironment()
        return self.callable(cfg, job_env)


def follow_slurm_job(job: Job) -> Generator[str, None, None]:
    drain = False
    with open(job.paths.stdout, "rb") as fo:
        with open(job.paths.stderr, "rb") as fe:
            while True:
                lo = fo.read().decode("UTF-8")

                if lo:
                    yield lo

                le = fe.read().decode("UTF-8")
                if le:
                    yield le

                time.sleep(0.1)


def slurm_launch(
    experiment: DictConfig,
    job: Callable[[DictConfig], None],
) -> None:
    hydra_state = Singleton.get_state()

    executor = SlurmExecutor(experiment.cluster.folder)
    executor.update_parameters(**experiment.cluster.job_params)

    job = executor.submit(SlurmSubmission(job), experiment, hydra_state)
    try:
        with yaspin(text="Waiting for jobs to start"):
            while job.state != "RUNNING" and not job.state.startswith("CANCELLED"):
                time.sleep(1)

        if job.state.startswith("CANCELLED"):
            print("Job canceled externally")
            exit()

        with yaspin(text=f"Attaching to master process logs: {job.paths.stdout.with_suffix('')}"):
            while not job.paths.stdout.exists():
                time.sleep(1)

        loglines = follow_slurm_job(job)
        for line in loglines:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        d = input("Quitting ... also cancel job? [Y/n]")

        if d.lower() == "y" or d == "":
            job.cancel()
            print("Job canceled")
        else:
            print("Job not canceled")
