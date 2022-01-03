from typing import Any, Iterable

import numpy as np
import numpy.typing as npt

from nemesys.hashing.minhashing.minhash import MinHash
from nemesys.modelling.routers.concatenation.concatenation_router import (
    ConcatenationRouter,
)

class MinHashConcatenationRouter(ConcatenationRouter):
    def __init__(self, minhash_instance: MinHash): ...
    @property
    def minhash_instance(self) -> MinHash: ...
    def concatenate(
        self, inputs: Iterable[Iterable[Iterable[Any]]]
    ) -> Iterable[Iterable[Any]]: ...
    def route(self, inputs: Iterable[Any]) -> npt.NDArray[np.uint64]:
        return self.concatenate(inputs=inputs)
