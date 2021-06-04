from pathlib import Path
from typing import Sequence, Tuple, Union

from torch import Tensor
from torchvision.io import write_png

from .serializer import Serializer


class ImageSerializer(Serializer):
    """
    TODO should these be upstreamed someplace?
    """

    def __init__(self, root_dir: Union[str, Path], pad: int = 5) -> None:
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.pad = pad

    def __call__(self, batch: Union[Tensor, Sequence[Tensor], Tuple[Union[Tensor, Sequence[Tensor]], Sequence[Path]]]) -> None:
        if isinstance(batch, tuple):
            images = batch[0]
            paths = batch[1]
        else:
            images = batch
            paths = None

        it = enumerate(images)

        for i, b in it:
            if paths is not None:
                p = paths[i]
                p = p.with_suffix(".png")
                out_path = self.root_dir / p
            else:
                idx = self.get_sync_index()
                out_path = (self.root_dir / str(idx).zfill(self.pad)).with_suffix(".png")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            write_png((b * 255).byte().cpu(), str(out_path))
