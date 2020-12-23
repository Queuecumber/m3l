from pathlib import Path

import pytorch_lightning as pl
from mme.data import JPEGQuantizedDataset, UnlabeledImageFolder
from mme.data.color_patch import ColorPatch
from mme.models import QGACCrab
from mme.models.qgac_crab import QGACCrab
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchjpeg.dct import Stats
from torchvision.transforms import ColorJitter, Compose, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor

train_transform = Compose(
    [
        RandomAffine(degrees=30, translate=(0.1, 0.5), scale=(0.8, 1.5), shear=(-25, 25, -25, 25), resample=Image.BILINEAR),
        RandomCrop(512, pad_if_needed=True),
        ColorJitter(0.25, 0.25, 0.25, 0.25),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ]
)

stats = Stats(Path("/private/home/mehrlich/compression-robust-pkg/src/compression_robust/stats/cstats.pt"))

div2k = UnlabeledImageFolder("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets/DIV2K", transform=train_transform)
flickr2k = UnlabeledImageFolder("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets/Flickr2K/Flickr2K_HR", transform=train_transform)
train_set = JPEGQuantizedDataset(ConcatDataset([div2k, flickr2k]), quality_range=(0, 100), stats=stats)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=10, pin_memory=True)

live1 = JPEGQuantizedDataset(UnlabeledImageFolder("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets/live1", transform=ToTensor()), quality_range=(10, 10), stats=stats)
val_loader = DataLoader(live1, batch_size=1, pin_memory=True, num_workers=10)

lod_dir = Path("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets")
dm = ColorPatch("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets/ColorPatchTrain", 
qgac = QGACCrab(stats)

trainer = pl.Trainer(gpus=1)
trainer.fit(qgac, train_loader, val_loader)
