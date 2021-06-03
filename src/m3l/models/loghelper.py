import logging
from typing import Optional

from pandas import DataFrame
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from torchvision.utils import make_grid

log = logging.getLogger(__name__)


class LogHelper:
    """
    TODO contribute to pytorch lightning logger class
    """

    def log_image(self, key: str, image: Tensor, caption: Optional[str] = None) -> None:
        if len(image.shape) == 3:
            im = make_grid(image)

        if isinstance(self.logger, WandbLogger):
            from wandb.data_types import Image

            self.logger.experiment.log({key: Image(image.movedim(0, 2).cpu().numpy(), caption=str(caption))})
        else:
            log.warn("Unsupported logger, ignoring log_image call")

    def log_table(self, key: str, table: DataFrame) -> None:
        if isinstance(self.logger, WandbLogger):
            from wandb.data_types import Table

            self.logger.experiment.log({key: Table(dataframe=table)})
