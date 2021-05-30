from pathlib import Path
from typing import Union

from torch import Tensor
from torchvision.io import write_png


class ImageSerializer:
    def __init__(self, root_dir: Union[str, Path]) -> None:
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir

    def __call__(self, image: Tensor, path: Path) -> None:
        out_path = (self.root_dir / path.name).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_png((image * 255).byte().cpu(), str(out_path))
