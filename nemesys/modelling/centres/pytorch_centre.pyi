from typing import Dict

import torch
import torch.nn

from nemesys.modelling.analysers.pytorch_analyser import PyTorchAnalyser
from nemesys.modelling.centres.centre import Centre
from nemesys.modelling.decoders.pytorch_decoder import PyTorchDecoder
from nemesys.modelling.encoders.pytorch_encoder import PyTorchEncoder
from nemesys.modelling.routers.pytorch_router import PyTorchRouter
from nemesys.modelling.stores.pytorch_store import PyTorchStore
from nemesys.modelling.synthesisers.pytorch_synthesiser import PyTorchSynthesiser

class PyTorchCentre(torch.nn.Module, Centre):
    def __init__(
        self,
        analyser: PyTorchAnalyser,
        encoder: PyTorchEncoder,
        decoder: PyTorchDecoder,
        router: PyTorchRouter,
        store: PyTorchStore,
        synthesiser: PyTorchSynthesiser,
    ): ...
    @property
    def analyser(self) -> PyTorchAnalyser: ...
    @property
    def decoder(self) -> PyTorchDecoder: ...
    @property
    def encoder(self) -> PyTorchEncoder: ...
    @property
    def router(self) -> PyTorchRouter: ...
    @property
    def store(self) -> PyTorchStore: ...
    @property
    def synthesiser(self) -> PyTorchSynthesiser: ...
    def process(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]: ...
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]: ...
