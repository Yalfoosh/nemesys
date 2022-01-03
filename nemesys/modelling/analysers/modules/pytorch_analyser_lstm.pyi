from typing import Dict, Iterable, List

import torch

class PyTorchAnalyserLSTM(torch.nn.Module):
    def __init__(
        self,
        class_names: Iterable[str],
        *args,
        **kwargs,
    ): ...
    @property
    def class_names(self) -> List[str]: ...
    def forward(self, inputs: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]: ...
