import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence, Union

import pytorch_lightning as pl
import torch.distributed
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchjpeg.dct import Stats
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import ColorJitter, Compose, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip

from .jpeg_quantized_dataset import JPEGQuantizedDataset, pad_coefficients_collate
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

    def __init__(
        self,
        root_dir: Union[str, Path],
        stats: Stats,
        train_batch_size: int,
        num_workers: int,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        samples_total: Optional[int] = 14400,
        cache_dir: Union[str, Path] = None,
    ) -> None:
        super().__init__()

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        self.root_dir = root_dir
        self.cache_dir = cache_dir

        self.stats = stats

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

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

                copytree_progress(self.root_dir / "BSDS500", self.cache_dir / "BSDS500", desc="Cache test data (2/3)", dirs_exist_ok=True)
                copytree_progress(self.root_dir / "ICB-RGB8", self.cache_dir / "ICB-RGB8", desc="Cache test data (3/3)", dirs_exist_ok=True)

            except Exception as e:
                log.warning(e)
                log.warning("Unable to copy to cache directory, using original as a fallback")
                self.cache_dir = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.cache_dir is not None:
            target = self.cache_dir
        else:
            target = self.root_dir

        if stage in ("fit", None):
            div2k = UnlabeledImageFolder(target / "DIV2K", transform=self.train_transforms)
            flickr2k = UnlabeledImageFolder(target / "Flickr2K" / "Flickr2K_HR", transform=self.train_transforms)
            self.patches = JPEGQuantizedDataset(ConcatDataset([div2k, flickr2k]), quality=(0, 100), stats=self.stats)

            self.live1 = JPEGQuantizedDataset(UnlabeledImageFolder(target / "live1", transform=ToTensor()), quality=10, stats=self.stats)

        if stage in ("test", None):
            self.live1 = [JPEGQuantizedDataset(UnlabeledImageFolder(target / "live1", transform=ToTensor()), quality=q, stats=self.stats) for q in range(10, 101, 10)]
            self.bsds = [JPEGQuantizedDataset(UnlabeledImageFolder(target / "BSDS500", transform=ToTensor()), quality=q, stats=self.stats) for q in range(10, 101, 10)]
            self.icb = [JPEGQuantizedDataset(UnlabeledImageFolder(target / "ICB-RGB8", transform=ToTensor()), quality=q, stats=self.stats) for q in range(10, 101, 10)]

    def train_dataloader(self) -> DataLoader:
        if self.samples_total is not None:
            sampler = RandomSampler(self.patches, replacement=True, num_samples=self.samples_total)
        elif torch.distributed.is_available():
            sampler = DistributedSampler(self.patches, shuffle=True)
        else:
            sampler = RandomSampler(self.patches)

        return DataLoader(self.patches, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        if torch.distributed.is_available():
            val_sampler = DistributedSampler(self.live1, shuffle=False)
        else:
            val_sampler = None

        return DataLoader(self.live1, batch_size=self.val_batch_size, num_workers=self.num_workers, pin_memory=True, sampler=val_sampler, collate_fn=pad_coefficients_collate)

    def test_dataloader(self) -> Sequence[DataLoader]:
        ds_seq = self.icb + self.live1 + self.bsds

        self.test_set_idx_map = {i: (["ICB", "Live-1", "BSDS500"][i // 10], i % 10 * 10 + 10) for i in range(len(ds_seq))}

        if torch.distributed.is_available():
            samplers = [DistributedSampler(d, shuffle=False) for d in ds_seq]
        else:
            samplers = [None for _ in ds_seq]

        dl_seq = [DataLoader(d, batch_size=self.test_batch_size, num_workers=self.num_workers, pin_memory=True, sampler=s, collate_fn=pad_coefficients_collate) for d, s in zip(ds_seq, samplers)]
        return dl_seq

    def teardown(self, stage: Optional[str] = None):
        if self.cache_dir is not None:
            log.info(f"Removing cache dir {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
