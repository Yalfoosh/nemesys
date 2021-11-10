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
        store: Store,
        analyser: Analyser,
        encoder: Encoder,
        decoder: Decoder,
        router: Router,
        synthesiser: Synthesiser,
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
    def analyser(self) -> Analyser:
        return self._analyser

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def router(self) -> Router:
        return self._router

    @property
    def store(self) -> Store:
        return self._store

    @property
    def synthesiser(self) -> Synthesiser:
        return self._synthesiser

    # endregion

    def process(self, inputs: Any) -> Any:
        raise NotImplementedError
