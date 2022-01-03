import torch

from nemesys.modelling.centres.centre import Centre


class PyTorchCentre(torch.nn.Module, Centre):
    def __init__(self, analyser, encoder, decoder, router, store, synthesiser):
        torch.nn.Module.__init__(self)
        Centre.__init__(self, analyser, decoder, encoder, router, store, synthesiser)

    def process(self, inputs):
        raise NotImplementedError

    def forward(self, inputs):
        return self._process(inputs=inputs)
