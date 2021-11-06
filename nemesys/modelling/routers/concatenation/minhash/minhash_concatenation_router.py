from typing import Any, Iterable

import numpy as np
import numpy.typing as npt

from nemesys.hashing.minhashing.minhash import MinHash
from nemesys.modelling.routers.concatenation.concatenation_router import (
    ConcatenationRouter,
)


class MinHashConcatenationRouter(ConcatenationRouter):
    def __init__(self, minhash_instance: MinHash):
        self._minhash_instance = minhash_instance

    @property
    def minhash_instance(self) -> MinHash:
        return self._minhash_instance

    def concatenate(self, inputs: Iterable[Any]) -> npt.NDArray[np.uint64]:
        minhashed_inputs = self.minhash_instance.get_minhash_many_eager(
            data_many=inputs
        )
        concatenated_output = np.concatenate(minhashed_inputs)

        return concatenated_output

    def route(self, inputs: Iterable[Any]) -> npt.NDArray[np.uint64]:
        return self.concatenate(inputs=inputs)
