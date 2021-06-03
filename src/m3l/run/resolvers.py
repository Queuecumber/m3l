from functools import reduce
from operator import mul
from pathlib import Path

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("m3l.modpath", lambda: Path(__file__).parent.absolute())
OmegaConf.register_new_resolver("m3l.mul", lambda *args: reduce(mul, args, 1))
OmegaConf.register_new_resolver("m3l.sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("m3l.get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver("m3l.get_class", hydra.utils.get_class)
