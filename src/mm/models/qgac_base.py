from typing import Any, Optional, Tuple

from torchtyping import TensorType

QGACTrainingBatch = Tuple[
    TensorType["batch", "channels":1, "height", "width"],
    Optional[TensorType["batch", "channels":2, "height/2", "width/2"]],
    TensorType["batch", "channels":1, "block_height":8, "block_width":8],
    Optional[TensorType["batch", "channels":1, "block_height":8, "block_width":8]],
    TensorType["batch", "channels":3, "height", "width"],
    TensorType["batch", "channel":2, "dimensions":2],
    Any,
]
