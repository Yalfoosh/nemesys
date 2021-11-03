from typing import Dict

import torch
import torch.nn

from nemesys.modelling.analysers.analyser import Analyser


class PyTorchAnalyser(Analyser, torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self._module = module

    # region Properties
    @property
    def module(self) -> torch.nn.Module:
        return self._module

    # endregion

    def analyse(self, inputs: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._module(inputs)

    def forward(self, inputs: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.analyse(inputs=inputs)
