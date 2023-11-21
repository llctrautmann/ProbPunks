import dataclasses
from typing import Optional

import torch

@dataclasses.dataclass
class Params:
    learning_rate: float = 3e-4
    beta: float = 0.5
    epochs: int = 100
    batch_size: int = 64
    device: str = "cuda:3" if torch.cuda.is_available() else "cpu"
    im_width: int = 128
    im_height: int = 128


hp = Params()