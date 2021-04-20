from pathlib import Path

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mm.modpath", lambda: Path(__file__).parent.absolute())
OmegaConf.register_new_resolver("mm.mul", lambda x, y: x * y)
