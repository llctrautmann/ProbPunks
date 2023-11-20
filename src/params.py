import dataclasses
from typing import Optional

import torch

@dataclasses.dataclass
class Params:
    learning_rate: float = 5e-4
    beta: float = 0.5
    epochs: int = 150
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    im_width: int = 256
    im_height: int = 256


hp = Params()