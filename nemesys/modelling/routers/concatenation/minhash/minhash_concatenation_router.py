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

    def concatenate(self, inputs: Iterable[Any]) -> Iterable[Any]:
        minhashed_inputs_list = self.minhash_instance.get_minhash_many_eager(
            data_many=inputs
        )
        concatenated_output = [
            minhashed_input
            for minhashed_inputs in minhashed_inputs_list
            for minhashed_input in minhashed_inputs
        ]

        return concatenated_output

    def route(self, inputs: Iterable[Any]) -> npt.NDArray[np.uint64]:
        return self.concatenate(inputs=inputs)
