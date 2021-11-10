from typing import Dict

import torch
import torch.nn

from modelling.analysers.pytorch_analyser import PyTorchAnalyser
from modelling.decoders.pytorch_decoder import PyTorchDecoder
from modelling.encoders.pytorch_encoder import PyTorchEncoder
from modelling.routers.pytorch_router import PyTorchRouter
from modelling.stores.pytorch_store import PyTorchStore
from modelling.synthesisers.pytorch_synthesiser import PyTorchSynthesiser


class PyTorchCentre(torch.nn.Module):
    def __init__(
        self,
        store: PyTorchStore,
        analyser: PyTorchAnalyser,
        encoder: PyTorchEncoder,
        decoder: PyTorchDecoder,
        router: PyTorchRouter,
        synthesiser: PyTorchSynthesiser,
    ):
        super().__init__()

        self._store = store
        self._analyser = analyser
        self._encoder = encoder
        self._decoder = decoder
        self._router = router
        self._synthesiser = synthesiser

    # region Properties
    @property
    def analyser(self) -> PyTorchAnalyser:
        return self._analyser

    @property
    def decoder(self) -> PyTorchDecoder:
        return self._decoder

    @property
    def encoder(self) -> PyTorchEncoder:
        return self._encoder

    @property
    def router(self) -> PyTorchRouter:
        return self._router

    @property
    def store(self) -> PyTorchStore:
        return self._store

    @property
    def synthesiser(self) -> PyTorchSynthesiser:
        return self._synthesiser

    # endregion

    def process(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._process(inputs=inputs)
