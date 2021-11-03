from typing import Any


from nemesys.modelling.stores.store import Store


class Decoder:
    def decode(self, inputs: Any, store: Store) -> Any:
        raise NotImplementedError
