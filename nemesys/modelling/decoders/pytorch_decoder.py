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

    def decode(self, store: Store) -> Dict[str, torch.Tensor]:
        return self._module(store)

    def forward(self, store: Store) -> Dict[str, torch.Tensor]:
        return self.decode(store=store)
