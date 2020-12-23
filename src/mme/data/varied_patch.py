from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import ConcatDataset
from torchjpeg.dct import Stats
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import ColorJitter, Compose, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip

from .jpeg_quantized_dataset import JPEGQuantizedDataset
from .unlabeled_image_folder import UnlabeledImageFolder


class VariedPatch(pl.LightningDataModule):
    """
    Varied patch dataset, extracts patches on-the-fly from the Flickr2k and DIV2k datasets using random affine and color jitter transformations. This creates
    so many patch combinations that it is recommended to sample with replacement and set a per-epoch maximum. Use `samples_total` to set this maximum (8000 by
    default) or set it to `None` to disable sampling with replacement.
    """

    def __init__(self, root_dir: Union[str, Path], stats: Stats, batch_size: int, num_workers: int, samples_total: Optional[int] = 8000) -> None:
        super().__init__()

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir

        self.stats = stats

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_total = 8000

        self.train_transforms = Compose(
            [
                RandomAffine(degrees=30, translate=(0.1, 0.5), scale=(0.8, 1.5), shear=(-25, 25, -25, 25), resample=Image.BILINEAR),
                RandomCrop(256, pad_if_needed=True),
                ColorJitter(0.25, 0.25, 0.25, 0.25),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:
            div2k = UnlabeledImageFolder(self.root_dir / "DIV2K", transform=self.train_transforms)
            flickr2k = UnlabeledImageFolder(self.root_dir / "Flickr2K" / "Flickr2K_HR", transform=self.train_transforms)
            self.patches = JPEGQuantizedDataset(ConcatDataset([div2k, flickr2k]), quality_range=(0, 100), stats=self.stats)

            self.live1 = JPEGQuantizedDataset(UnlabeledImageFolder(self.root_dir / "live1", transform=ToTensor()), quality_range=(10, 10), stats=self.stats)

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        if self.samples_total is not None:
            with_replacement_sampler = RandomSampler(self.patches, replacement=True, num_samples=self.samples_total)
        else:
            with_replacement_sampler = None

        return DataLoader(self.patches, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, sampler=with_replacement_sampler)

    def val_dataloader(self):
        return DataLoader(self.live1, batch_size=1, num_workers=self.num_workers, pin_memory=True)
