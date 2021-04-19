import logging
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchjpeg.dct import Stats
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import ColorJitter, Compose, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip

from .jpeg_quantized_dataset import JPEGQuantizedDataset
from .unlabeled_image_folder import UnlabeledImageFolder
from .utils import copytree_progress

log = logging.getLogger(__name__)


class VariedPatch(pl.LightningDataModule):
    """
    Varied patch dataset, extracts patches on-the-fly from the Flickr2k and DIV2k datasets using random affine and color jitter transformations. This creates
    so many patch combinations that it is recommended to sample with replacement and set a per-epoch maximum. Use `samples_total` to set this maximum (8000 by
    default) or set it to `None` to disable sampling with replacement. This will also make a DistributedSampler for the val set so you can safely set
    `replace_sampler_ddp` to False in the pytorch lightning Trainer.
    """

    def __init__(self, root_dir: Union[str, Path], stats: Stats, batch_size: int, num_workers: int, samples_total: Optional[int] = 14400, cache_dir: Union[str, Path] = None) -> None:
        super().__init__()

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        self.root_dir = root_dir
        self.cache_dir = cache_dir

        self.stats = stats

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_total = samples_total

        self.train_transforms = Compose(
            [
                # RandomAffine(degrees=15, translate=(0.1, 0.3), scale=(0.7, 1.3), shear=(-15, 15, -15, 15), resample=Image.BILINEAR),
                RandomCrop(size=256, pad_if_needed=True),
                # ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
            ]
        )

    def prepare_data(self) -> None:
        if self.cache_dir is not None:
            try:
                copytree_progress(self.root_dir / "DIV2K", self.cache_dir / "DIV2K", desc="Cache training data (1/2)", dirs_exist_ok=True)
                copytree_progress(self.root_dir / "Flickr2K" / "Flickr2K_HR", self.cache_dir / "Flickr2K" / "Flickr2K_HR", desc="Cache training data (2/2)", dirs_exist_ok=True)
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

        if stage == "fit" or stage is None:
            div2k = UnlabeledImageFolder(target / "DIV2K", transform=self.train_transforms)
            flickr2k = UnlabeledImageFolder(target / "Flickr2K" / "Flickr2K_HR", transform=self.train_transforms)
            self.patches = JPEGQuantizedDataset(ConcatDataset([div2k, flickr2k]), quality=(0, 100), stats=self.stats)

            self.live1 = JPEGQuantizedDataset(UnlabeledImageFolder(target / "live1", transform=ToTensor()), quality=10, stats=self.stats)

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        if self.samples_total is not None:
            with_replacement_sampler = RandomSampler(self.patches, replacement=True, num_samples=self.samples_total)
        else:
            with_replacement_sampler = None

        return DataLoader(self.patches, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, sampler=with_replacement_sampler)

    def val_dataloader(self):
        # TODO make this work in non-distributed mode
        if self.samples_total is not None:
            val_sampler = DistributedSampler(self.live1, shuffle=False)

        return DataLoader(self.live1, batch_size=1, num_workers=self.num_workers, pin_memory=True, sampler=val_sampler)
