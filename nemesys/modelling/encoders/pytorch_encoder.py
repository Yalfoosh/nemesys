import torch

from nemesys.modelling.encoders.encoder import Encoder


class PyTorchEncoder(torch.nn.Module, Encoder):
    def __init__(self, module):
        torch.nn.Module.__init__()
        Encoder.__init__()

        self._module = module

    # region Properties
    @property
    def module(self):
        return self._module

    # endregion

    def encode(self, inputs):
        return self._module(inputs)

    def forward(self, inputs):
        return self.encode(inputs=inputs)
