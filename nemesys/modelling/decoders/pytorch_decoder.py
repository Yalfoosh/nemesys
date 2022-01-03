import torch
import torch.nn

from nemesys.modelling.decoders.decoder import Decoder


class PyTorchDecoder(torch.nn.Module, Decoder):
    def __init__(self, module):
        torch.nn.Module.__init__(self)
        Decoder.__init__()

        self._module = module

    # region Properties
    @property
    def module(self):
        return self._module

    # endregion

    def decode(self, store):
        return self._module(store)

    def forward(self, store):
        return self.decode(store=store)
