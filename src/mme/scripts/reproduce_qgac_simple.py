"""
This script is designed to be as simple as possible while reproducing the procedure exactly from the original
QGAC paper (Quantization Guided JPEG Artifact Correction). The procedure has some quirks to it. The high
level view is:

1. The training set used pre-extracted patches, appx 30000 of them. 
2. The epoch length is artificially set to 1000 batches, so appx 30 artificial epochs per real epoch.  
3. Validation happens every 25 batches
4. The optimizer is Adam with default settings and LR starting at 1e-4
5. The LR decays with a cosine annealing scheduler to 1e-6. The step happens at the end of each artificial epoch
6. Training runs for 100 total artificial epochs (3 and change real epochs).
7. After every validation, a checkpoint is saved (original QGAC retained all of these checkpoints causing the checkpoint dir to explode in size, we only keep the most recent one.)
8. The training happens in batches of 32 (original QGAC used 8 GPUs, so batches of 4 per GPU, we use a single GPU.)

This is a little tricky to get right using PL. The optimizer and LR scheduler details are in the QGAC class, a LightningModule (the scheduler decaying after 1000 batches is here). 
The dataset is in the ColorPatch class, a LightningDataModule. The rest of the procedure is in this file, but it may be hard to see. We need to override the checkpointer to checkpoint 
every 25 steps, we need to set max_steps in the trainer to 100 * 1000 steps total (100 artifical epochs where each artificial epoch is 1000 batches), and we need to set val_check_interval
to 25 to validate every 25 batches.

"""
import os
from pathlib import Path

import pytorch_lightning as pl
from mme.data import ColorPatch
from mme.models import QGAC
from pytorch_lightning.callbacks import ModelCheckpoint
from torchjpeg.dct import Stats


# See https://github.com/PyTorchLightning/pytorch-lightning/issues/2534#issuecomment-674582085
class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self, save_step_frequency):
        """
        Args:
            save_step_frequency: how often to save in steps
        """
        self.save_step_frequency = save_step_frequency

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, "checkpoint_latest.ckpt")  # Unlike the original QGAC, we only save a single checkpoint
            trainer.save_checkpoint(ckpt_path)


def main():
    stats = Stats(Path("/private/home/mehrlich/compression-robust-pkg/src/compression_robust/stats/cstats.pt"))

    lid_dir = Path("/checkpoint/mehrlich/lossless_image_datasets/lossless_image_datasets")
    dm = ColorPatch(root_dir=lid_dir, stats=stats, batch_size=32, num_workers=10)

    qgac = QGAC(stats=stats)  # Details of the LR scheduler and optimizer are in this class

    trainer = pl.Trainer(gpus=1, max_steps=100 * 1000, val_check_interval=25, callbacks=[CheckpointEveryNSteps(25)])  # artifical epoch len is 1000
    trainer.fit(qgac, dm)


if __name__ == "__main__":
    main()
