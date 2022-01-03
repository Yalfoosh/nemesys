class Centre:
    def __init__(self, analyser, encoder, decoder, router, store, synthesiser):
        self._analyser = analyser
        self._encoder = encoder
        self._decoder = decoder
        self._router = router
        self._store = store
        self._synthesiser = synthesiser

    # region Properties
    @property
    def analyser(self):
        return self._analyser

    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def router(self):
        return self._router

    @property
    def store(self):
        return self._store

    @property
    def synthesiser(self):
        return self._synthesiser

    # endregion

    def process(self, inputs):
        raise NotImplementedError
