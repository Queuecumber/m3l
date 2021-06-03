from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm


class M3LProgress(ProgressBar):
    def init_predict_tqdm(self) -> tqdm:
        bar = super().init_predict_tqdm()
        bar.set_description("Correcting")
        return bar
