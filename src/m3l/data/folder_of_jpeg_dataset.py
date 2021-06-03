from pathlib import Path
from typing import Sequence, Tuple, Union

import torch
import torchjpeg.codec
from torch import Tensor
from torchjpeg.dct import Stats, deblockify, normalize
from torchjpeg.quantization.ijg import quantization_max

from .imagelist import ImageList


class FolderOfJpegDataset(torch.utils.data.Dataset):
    def __init__(self, path: Union[str, Path], stats: Stats, extensions: Sequence[str] = [".jpg", ".jpeg", ".JPEG"]):
        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.stats = stats

        if path.is_dir():
            self.images = list(filter(lambda p: p.suffix in extensions, path.glob("**/*")))
        else:
            self.images = [path]

    def __len__(self):
        return len(self.images)

    def __dequantize_channel(self, channel, quantization):
        dequantized_dct = channel.float() * quantization
        dequantized_dct = dequantized_dct.view(1, 1, dequantized_dct.shape[1] * dequantized_dct.shape[2], 8, 8)
        dequantized_dct = deblockify(dequantized_dct, (channel.shape[1] * 8, channel.shape[2] * 8))

        return dequantized_dct

    def __getitem__(self, idx):
        image = self.images[idx]

        dim, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(str(image))
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

        return y_dequantized.squeeze(0), cbcr_dequantized.squeeze(0), y_q.unsqueeze(0), c_q.unsqueeze(0), image, dim[0]

    @staticmethod
    def collate(batch_list: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor, Path, Tensor]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Sequence[Path], Tensor]:
        y_coefs = []
        cbcr_coefs = []

        yqs = torch.stack([b[2] for b in batch_list])
        cqs = torch.stack([b[3] for b in batch_list])
        sizes = torch.stack([b[5] for b in batch_list])

        paths = [b[4] for b in batch_list]

        for b in batch_list:
            y_coefs.append(b[0])
            cbcr_coefs.append(b[1])

        y_coefs = ImageList.from_tensors(y_coefs).tensor
        cbcr_coefs = ImageList.from_tensors(cbcr_coefs).tensor

        return y_coefs, cbcr_coefs, yqs, cqs, paths, sizes
