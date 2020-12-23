from typing import Optional, Tuple

from torch import Tensor

QGACTrainingBatch = Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor], Tensor, Tensor]
