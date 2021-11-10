from typing import Dict

import torch
import torch.nn
import torch.nn.functional


class PyTorchEncoderLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        content_key: str = "content",
    ):
        super().__init__()

        self._linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self._content_key = content_key

    @property
    def content_key(self) -> str:
        return self._content_key

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = inputs[self._content_key]

        return {"content": self._linear(inputs)}
