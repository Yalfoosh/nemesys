import torch

from nemesys.modelling.analysers.analyser import Analyser


class PyTorchAnalyser(Analyser, torch.nn.Module):
    def __init__(self, module):
        super().__init__()

        self._module = module

    # region Properties
    @property
    def module(self):
        return self._module

    # endregion

    def analyse(self, inputs):
        return self._module(inputs)

    def forward(self, inputs):
        return self.analyse(inputs=inputs)
