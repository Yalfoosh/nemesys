from typing import Dict

import torch
import torch.nn
import torch.nn.functional


class PyTorchSynthesiserLinear(torch.nn.Module):
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

    def forward(
        self, inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # (n, batch_size, ..., in_features)
        inputs = torch.stack(
            tuple(x[self._content_key] for x in inputs.values()), dim=0
        )
        # (batch_size, ..., in_features)
        inputs = torch.sum(inputs, dim=0)

        return {"content": self._linear(inputs)}
