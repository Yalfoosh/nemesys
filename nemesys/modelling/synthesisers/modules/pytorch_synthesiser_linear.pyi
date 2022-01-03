from typing import Dict

import torch

class PyTorchSynthesiserLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        content_key: str,
    ): ...
    @property
    def content_key(self) -> str: ...
    def forward(
        self, inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]: ...
