from typing import Dict

import torch
import torch.nn

from nemesys.modelling.decoders.decoder import Decoder
from nemesys.modelling.stores.store import Store


class PyTorchDecoder(Decoder, torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self._module = module

    # region Properties
    @property
    def module(self) -> torch.nn.Module:
        return self._module

    # endregion

    def decode(
        self, inputs: Dict[str, Dict[str, torch.Tensor]], store: Store
    ) -> Dict[str, torch.Tensor]:
        return self._module(inputs)

    def forward(
        self, inputs: Dict[str, Dict[str, torch.Tensor]], store: Store
    ) -> Dict[str, torch.Tensor]:
        return self.decode(inputs=inputs, store=store)
