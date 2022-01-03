from typing import Iterable

import torch

class PyTorchRouter:
    def route(self, inputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]: ...
