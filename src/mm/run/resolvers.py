from functools import reduce
from operator import mul
from pathlib import Path

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mm.modpath", lambda: Path(__file__).parent.absolute())
OmegaConf.register_new_resolver("mm.mul", lambda *args: reduce(args, mul, 1))
OmegaConf.register_new_resolver("mm.sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("mm.get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver("mm.get_class", hydra.utils.get_class)
