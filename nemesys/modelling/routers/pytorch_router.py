from typing import Any, Iterable

import torch


class PyTorchRouter:
    def route(self, inputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        raise NotImplementedError
