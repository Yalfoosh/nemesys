from typing import Dict

import torch

from nemesys.modelling.synthesisers.synthesiser import Synthesiser

class PyTorchSynthesiser(torch.nn.Module, Synthesiser):
    def __init__(self, module): ...
    @property
    def module(self) -> torch.nn.Module: ...
    def synthesise(
        self, components: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]: ...
    def forward(
        self, components: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]: ...
