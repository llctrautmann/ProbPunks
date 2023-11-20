import dataclasses
from typing import Optional

import torch

@dataclasses.dataclass
class Params:
    learning_rate: float = 3e-4
    beta: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    im_width: int = 256
    im_height: int = 256


hp = Params()