from typing import Dict

import torch
import torch.nn

from nemesys.modelling.encoders.encoder import Encoder


class PyTorchEncoder(Encoder, torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self._module = module

    # region Properties
    @property
    def module(self) -> torch.nn.Module:
        return self._module

    # endregion

    def encode(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._module(inputs)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.encode(inputs=inputs)
