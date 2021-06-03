import logging
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchjpeg.dct import Stats
from torchvision.transforms import ToTensor

from .jpeg_quantized_dataset import JPEGQuantizedDataset
from .unlabeled_image_folder import UnlabeledImageFolder
from .utils import copytree_progress

log = logging.getLogger(__name__)


class ColorPatch(pl.LightningDataModule):
    """
    Color patch dataset used in Quantization Guided JPEG Artifact Correction, uses pre-extracted color patches for train compressed in [10, 100] and live1 for val compressed at quality 10. Patches are
    pre-extracted
    """

    def __init__(self, root_dir: Union[str, Path], stats: Stats, batch_size: int, num_workers: int, cache_dir: Union[str, Path] = None) -> None:
        super().__init__()

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        self.root_dir = root_dir

        self.stats = stats

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        if self.cache_dir is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                copytree_progress(self.root_dir / "ColorPatchTrain", self.cache_dir / "ColorPatchTrain", desc="Cache training data", dirs_exist_ok=True)
                copytree_progress(self.root_dir / "live1", self.cache_dir / "live1", desc="Cache val data", dirs_exist_ok=True)
            except Exception as e:
                log.warning(e)
                log.warning("Unable to copy to cache directory, using original as a fallback")
                self.cache_dir = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.cache_dir is not None:
            target = self.cache_dir
        else:
            target = self.root_dir

        self.patches = JPEGQuantizedDataset(UnlabeledImageFolder(target / "ColorPatchTrain", transform=ToTensor()), quality=(10, 100, 10), stats=self.stats, deterministic_quality=True)
        self.live1 = JPEGQuantizedDataset(UnlabeledImageFolder(target / "live1", transform=ToTensor()), quality=10, stats=self.stats)

    def train_dataloader(self):
        return DataLoader(self.patches, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.live1, batch_size=1, num_workers=self.num_workers, pin_memory=True)
