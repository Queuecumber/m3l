import os
from glob import glob
from pathlib import Path
from typing import Sequence, Union

import torch
import torchjpeg.codec
from torchjpeg.dct import Stats, deblockify, normalize
from torchjpeg.quantization.ijg import quantization_max


class FolderOfJpegDataset(torch.utils.data.Dataset):
    def __init__(self, path: Union[str, Path], stats: Stats, extensions: Sequence[str] = [".jpg"]):
        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.images = list(filter(lambda p: p.suffix in extensions, path.glob("**/*")))
        self.stats = stats

    def __len__(self):
        return len(self.images)

    def __dequantize_channel(self, channel, quantization):
        dequantized_dct = channel.float() * quantization
        dequantized_dct = dequantized_dct.view(1, dequantized_dct.shape[1] * dequantized_dct.shape[2], 8, 8)
        dequantized_dct = deblockify(dequantized_dct, 1, (channel.shape[1] * 8, channel.shape[2] * 8))

        return dequantized_dct

    def __getitem__(self, idx):
        image = self.images[idx]

        dim, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(image)
        quantization = quantization.float()

        y_q = quantization[0]

        y_dequantized = self.__dequantize_channel(Y_coefficients, y_q)
        y_q /= quantization_max
        y_dequantized = normalize(y_dequantized, self.stats, channel="y")

        if CbCr_coefficients is not None:
            c_q = quantization[1]  # Assume same quantization for cb and cr

            cb_dequantized = self.__dequantize_channel(CbCr_coefficients[0:1], c_q)
            cr_dequantized = self.__dequantize_channel(CbCr_coefficients[1:2], c_q)

            c_q = c_q / quantization_max

            cb_dequantized = normalize(cb_dequantized, self.stats, channel="cb")
            cr_dequantized = normalize(cr_dequantized, self.stats, channel="cr")

            cbcr_dequantized = torch.cat([cb_dequantized, cr_dequantized], dim=1)
        else:
            cbcr_dequantized = torch.empty(0)
            c_q = torch.empty(0)

        return y_dequantized.squeeze(0), cbcr_dequantized.squeeze(0), y_q.unsqueeze(0), c_q.unsqueeze(0), image, dim
