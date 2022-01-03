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

    def concatenate(
        self, inputs: Iterable[Iterable[Iterable[Any]]]
    ) -> Iterable[Iterable[Any]]:
        # inputs: (n, batch, x)
        outputs = list()

        for entry in inputs:
            batches = list()

            for batch in entry:
                batches.append(
                    self.minhash_instance.get_minhash_batch_eager(data_batch=batch)
                )

            outputs.append(np.array(batches))

        # outputs: (n, batch, n_perm)
        outputs = np.array(outputs, dtype=np.uint64)

        # outputs: (batch, n * n_perm)
        outputs = np.concatenate(outputs, axis=1)

        return outputs

    def route(self, inputs: Iterable[Any]):  # -> npt.NDArray[np.uint64]:
        return self.concatenate(inputs=inputs)
