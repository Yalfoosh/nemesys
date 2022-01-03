from typing import Any

from modelling.analysers.analyser import Analyser
from modelling.decoders.decoder import Decoder
from modelling.encoders.encoder import Encoder
from modelling.routers.router import Router
from modelling.stores.store import Store
from modelling.synthesisers.synthesiser import Synthesiser

class Centre:
    def __init__(
        self,
        analyser: Analyser,
        encoder: Encoder,
        decoder: Decoder,
        router: Router,
        store: Store,
        synthesiser: Synthesiser,
    ): ...
    @property
    def analyser(self) -> Analyser: ...
    @property
    def decoder(self) -> Decoder: ...
    @property
    def encoder(self) -> Encoder: ...
    @property
    def router(self) -> Router: ...
    @property
    def store(self) -> Store: ...
    @property
    def synthesiser(self) -> Synthesiser: ...
    def process(self, inputs: Any) -> Any: ...
