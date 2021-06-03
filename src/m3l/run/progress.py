from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm


class M3LProgress(ProgressBar):
    """
    TODO I dont think we can get rid of this (or we just live with it saying Predicting instead of Correcting)
    """

    def init_predict_tqdm(self) -> tqdm:
        bar = super().init_predict_tqdm()
        bar.set_description("Correcting")
        return bar
