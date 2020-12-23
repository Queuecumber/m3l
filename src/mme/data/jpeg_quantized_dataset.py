from random import randint
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch.functional import Tensor
from torch.utils.data import Dataset
from torchjpeg.codec import quantize_at_quality
from torchjpeg.dct import Stats, deblockify, images_to_batch, normalize
from torchjpeg.quantization.ijg import quantization_max


class JPEGQuantizedDataset(Dataset):
    r"""
    Wraps an arbitrary image dataset to return JPEG quantized versions of the image. The amount quantization is defined using IJG quality settings.
    If the underlying dataset returns a sequence, the first element of the sequence is taken the be the image which is quantized and the remaining
    elements are returned as the last element of the batch. If the underlying dataset returns a mapping, set `image_key` to the key of the image
    to be quantized. The original dictionary, including the image before quantization, will be returned as the last element of the batch.

    Since the primary return values are all DCT coefficients, padding will be added to the images to make them an even multiple of the MCU. Following JPEG
    conventions this is replicate padding added to the bottom and right edges. The original size of the image is returned so that the images can be
    correctly cropped after processing.

    The format returned by this dataset is:

    Y Channel Coefficients, CbCr Coefficients, Y Quantization Matrix, CbCr Quantization Matrix, Pre-quantization YCbCr Coefficients, Original Image Size, Optional rest of the batch from the underlying dataset

    If the image is grayscale, the CbCr coefficients and quantization matrix will be an empty tensor, if the underlying dataset returns an image with no additional data, the final return value will be an empty tensor.
    This is to avoid issues with the default collate function, an empty tensor is one initialized with 0 size using :py:func:`torch.empty`. It is detectable as `tensor.numel() == 0`.

    Args:
        data (:py:class:`torch.utils.data.Dataset`): The dataset to wrap
        quality_range (tuple of two ints): The quality range to draw from, inclusive on both ends
        stats (:py:clas:`torchjpeg.dct.Stats`): Statstics to use for per-frequency per-channel coefficient normalization
        mcu (int): The size of the minimum coded unit, use 16 for 4:2:0 chroma subsampling.
        image_key (optional str): The key to use to extract the image from a dataset which returns a mapping.
    """

    def __init__(self, dataset: Dataset, quality_range: Tuple[int, int], stats: Stats, mcu: int = 16, image_key: Optional[str] = None) -> None:
        self.dataset = dataset
        self.quality_range = quality_range
        self.stats = stats
        self.mcu = mcu
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __dequantize_channel(self, channel, quantization):
        dequantized_dct = channel.float() * quantization
        dequantized_dct = dequantized_dct.view(1, 1, dequantized_dct.shape[1] * dequantized_dct.shape[2], 8, 8)
        dequantized_dct = deblockify(dequantized_dct, (channel.shape[1] * 8, channel.shape[2] * 8))

        return dequantized_dct

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Any]:
        image = self.dataset[idx]
        labels = torch.empty(0)

        if isinstance(image, Sequence):
            labels = image[1:]
            image = image[0]
        elif isinstance(image, Mapping):
            labels = image
            image = image[self.dict_key]

        quality = randint(*self.quality_range)

        s = torch.Tensor(list(image.shape))
        p = (torch.ceil(s / self.mcu) * self.mcu - s).long()
        image = torch.nn.functional.pad(image.unsqueeze(0), [0, p[2], 0, p[1]], "replicate").squeeze(0)

        (
            _,
            quantization,
            Y_coefficients,
            CbCr_coefficients,
        ) = quantize_at_quality(image, quality)

        groundtruth_dct = images_to_batch(image.unsqueeze(0), self.stats).squeeze(0)  # This *has* to be after quantization

        quantization = quantization.float()

        y_q = quantization[0]

        y_dequantized = self.__dequantize_channel(Y_coefficients, y_q)
        y_q = y_q / quantization_max
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

        return (
            y_dequantized.squeeze(0),
            cbcr_dequantized.squeeze(0),
            y_q.unsqueeze(0),
            c_q.unsqueeze(0),
            groundtruth_dct,
            s.long(),
            labels,
        )


# def pad_coefficients_collate(batch_list):
#     y_coefs = []
#     cb_coefs = []
#     cr_coefs = []
#     gt_coefs = []

#     yqs = torch.stack([b[3] for b in batch_list])
#     cqs = torch.stack([b[4] for b in batch_list])
#     sizes = torch.stack([b[6] for b in batch_list])

#     labels = [b[7] for b in batch_list]

#     for b in batch_list:
#         y_coefs.append(b[0])
#         cb_coefs.append(b[1])
#         cr_coefs.append(b[2])
#         gt_coefs.append(b[5])

#     y_coefs = ImageList.from_tensors(y_coefs).tensor
#     cb_coefs = ImageList.from_tensors(cb_coefs).tensor
#     cr_coefs = ImageList.from_tensors(cr_coefs).tensor
#     gt_coefs = ImageList.from_tensors(gt_coefs).tensor

#     return y_coefs, cb_coefs, cr_coefs, yqs, cqs, gt_coefs, sizes, labels
