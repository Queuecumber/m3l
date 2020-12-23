from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchjpeg.dct import Stats
from torchvision.transforms import ToTensor

from .jpeg_quantized_dataset import JPEGQuantizedDataset
from .unlabeled_image_folder import UnlabeledImageFolder


class VariedPatch(pl.LightningDataModule):
    """
    Varied patch dataset, extracts patches on-the-fly from the Flickr2k and DIV2k datasets using random affine and color jitter transformations. This creates
    so many patch combinations that it is recommended to sample with replacement and set a per-epoch maximum.
    """

    def __init__(self, patch_dir: Union[str, Path], live1_dir: Union[str, Path], stats: Stats, batch_size: int, num_workers: int) -> None:
        super().__init__()

        for p in (patch_dir, live1_dir):
            if isinstance(p, str):
                p = Path(p)

        self.patch_dir = patch_dir
        self.live1_dir = live1_dir

        self.stats = stats

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.patches = JPEGQuantizedDataset(UnlabeledImageFolder(self.patch_dir, transform=ToTensor()), quality_range=(10, 100), stats=self.stats)
        self.live1 = JPEGQuantizedDataset(UnlabeledImageFolder(self.live1_dir, transform=ToTensor()), quality_range=(10, 10), stats=self.stats)

    def train_dataloader(self):
        return DataLoader(self.patches, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.live1, batch_size=1, num_workers=self.num_workers, pin_memory=True)
