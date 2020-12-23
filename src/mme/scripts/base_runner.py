import asyncio
import os
import pathlib
import time
import warnings
from enum import Enum
from typing import Callable, Generator, Optional

import hydra
import torch
from hydra.core.singleton import Singleton
from omegaconf import DictConfig, OmegaConf
from submitit import Job, JobEnvironment
from submitit.helpers import DelayedSubmission
from yaspin import yaspin

OmegaConf.register_resolver("function", lambda x: hydra.utils.get_method(x))
OmegaConf.register_resolver("modpath", lambda: pathlib.Path(__file__).parent.absolute())


class CheckpointBehavior(Enum):
    RESUME = "resume"
    RESTART = "restart"
    NONE = "none"


class SingleJob:
    def __init__(self, callable: Callable[[DictConfig, JobEnvironment], None], port: int = 32017, checkpoint_behavior: CheckpointBehavior = CheckpointBehavior.RESUME) -> None:
        self.callable = callable
        self.port = port
        self.checkpoint_behavior = checkpoint_behavior

    def checkpoint(self, cfg: DictConfig, hydra_state: Singleton) -> Optional[DelayedSubmission]:
        if self.checkpoint_behavior == CheckpointBehavior.RESUME:
            cfg.checkpoint.resume = True

        if self.checkpoint_behavior != CheckpointBehavior.NONE:
            return DelayedSubmission(SingleJob(self.callable), cfg, hydra_state)

        return None

    def __call__(self, cfg: DictConfig, hydra_state: Singleton) -> None:
        Singleton.set_state(hydra_state)
        job_env = JobEnvironment()

        torch.distributed.init_process_group(
            backend="NCCL",
            init_method=f"tcp://{job_env.hostnames[0]}:{self.port}",
            rank=job_env.global_rank,
            world_size=job_env.num_tasks,
        )
        torch.cuda.set_device(job_env.local_rank)

        with warnings.catch_warnings():
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            warnings.filterwarnings("ignore")
            return self.callable(cfg, job_env)


async def follow_job_log(job: Job) -> Generator[str, None, None]:
    draining = False
    with open(job.paths.stdout, "rb") as fo:
        with open(job.paths.stderr, "rb") as fe:
            while True:
                lo = fo.read().decode("UTF-8")

                if lo:
                    yield lo

                le = fe.read().decode("UTF-8")
                if le:
                    yield le

                if job.state not in ["RUNNING", "PENDING", "REQUEUED"] and not le and not lo:
                    if draining:
                        break

                    draining = True
                    await asyncio.sleep(1)
                else:
                    await asyncio.sleep(0.1)


async def follow_impl(job: Job) -> None:
    try:
        with yaspin(text="Waiting for jobs to start"):
            while job.state != "RUNNING" and not job.state.startswith("CANCELLED"):
                await asyncio.sleep(1)

        if job.state.startswith("CANCELLED"):
            print("Job canceled externally")
            exit()

        with yaspin(text=f"Attaching to master process logs: {job.paths.stdout.with_suffix('')}"):
            while not job.paths.stdout.exists():
                await asyncio.sleep(1)

        loglines = await follow_job_log(job)
        async for line in loglines:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        d = input("Quitting ... also cancel job? [Y/n]")

        if d.lower() == "y" or d == "":
            job.cancel()
            print("Job canceled")
        else:
            print("Job not canceled")


def launcher(
    cfg: DictConfig,
    callable: Callable[[DictConfig, JobEnvironment], None],
    follow=True,
    port: int = 32017,
    env_override: Optional[DictConfig] = None,
    checkpoint_behavior: CheckpointBehavior = CheckpointBehavior.RESUME,
) -> Optional[Job]:
    hydra_state = Singleton.get_state()

    if env_override is None:
        env_override = cfg.env

    executor = hydra.utils.instantiate(env_override.executor)
    executor.update_parameters(**env_override.params)

    job = executor.submit(SingleJob(callable, port, checkpoint_behavior), cfg, hydra_state)

    if follow:
        asyncio.run(follow_impl(job))
    else:
        return job
