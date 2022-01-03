import torch

from nemesys.modelling.synthesisers.synthesiser import Synthesiser


class PyTorchSynthesiser(torch.nn.Module, Synthesiser):
    def __init__(self, module):
        torch.nn.Module.__init__()
        Synthesiser.__init__()

        self._module = module

    # region Properties
    @property
    def module(self):
        return self._module

    # endregion

    def synthesise(self, components):
        return self._module(components)

    def forward(self, components):
        return self.synthesise(components=components)
