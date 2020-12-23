from pathlib import Path

import pytorch_lightning as pl
from mme.data import VariedPatch
from mme.models import QGACCrab
from torchjpeg.dct import Stats

stats = Stats(Path("/private/home/mehrlich/compression-robust-pkg/src/compression_robust/stats/cstats.pt"))

lod_dir = Path("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets")
dm = VariedPatch(root_dir=lod_dir, stats=stats, batch_size=128, num_workers=10)
qgac = QGACCrab(stats=stats)

trainer = pl.Trainer(gpus=2, replace_sampler_ddp=False)
trainer.fit(qgac, dm)
