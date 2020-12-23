from pathlib import Path

import pytorch_lightning as pl
from mme.data import ColorPatch
from mme.models import QGAC
from torchjpeg.dct import Stats


def main():
    stats = Stats(Path("/private/home/mehrlich/compression-robust-pkg/src/compression_robust/stats/cstats.pt"))

    lid_dir = Path("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets")
    dm = ColorPatch(root_dir=lid_dir, stats=stats, batch_size=32, num_workers=10)
    qgac = QGAC(stats=stats)

    trainer = pl.Trainer(gpus=1, max_steps=100 * 1000, val_check_interval=1000)  # artifical epoch len is 1000
    trainer.fit(qgac, dm)


if __name__ == "__main__":
    main()
