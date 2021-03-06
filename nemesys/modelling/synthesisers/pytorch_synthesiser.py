from typing import Dict

import torch
import torch.nn

from nemesys.modelling.synthesisers.synthesiser import Synthesiser


class PyTorchSynthesiser(Synthesiser, torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self._module = module

    # region Properties
    @property
    def module(self) -> torch.nn.Module:
        return self._module

    # endregion

    def synthesise(
        self, components: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return self._module(components)

    def forward(
        self, components: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return self.synthesise(components=components)
