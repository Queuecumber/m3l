import os
from glob import glob
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
from PIL import Image


class UnlabeledImageFolder(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        extensions: Sequence[str] = [".bmp", ".png", ".jpg", ".ppm", ".pgm"],
        transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.images = list(filter(lambda p: p.suffix in extensions, path.glob("**/*")))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Any:
        im_path = self.images[idx]

        with open(im_path, "rb") as f:
            im = Image.open(f)
            im.load()

        if self.transform is not None:
            im = self.transform(im)

        return im
