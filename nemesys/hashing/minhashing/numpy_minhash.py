from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import numpy.typing as npt

from nemesys.hashing.minhashing.minhash import MinHash
from nemesys.utils.conversions.numpy_conversion import NumPyConversion


class NumPyMinHash(MinHash):
    def __init__(
        self,
        n_permutations: int,
        seed: int,
        preprocess_function,  #: Callable[[Any], npt.NDArray[np.uint64]],
        prime: Optional[Union[np.uint64, int]] = None,
        base_state: Optional[Iterable[Union[np.uint32, int]]] = None,
        bound: Optional[Union[np.uint64, int]] = None,
    ):
        if prime is None:
            prime = self.default_prime
        if base_state is None:
            self._n_permutations = n_permutations  # TODO: FIX THIS!
            base_state = self.default_base_state
        if bound is None:
            bound = self.default_bound

        super().__init__(
            n_permutations=n_permutations,
            seed=seed,
            preprocess_function=preprocess_function,
            prime=NumPyConversion.to_array(content=prime, dtype=np.uint64),
            base_state=NumPyConversion.to_array(content=base_state, dtype=np.uint32),
            bound=NumPyConversion.to_array(content=bound, dtype=np.uint64),
        )

        generator = np.random.RandomState(seed=self._seed)
        self._a = generator.randint(
            low=1,
            high=self.prime,
            size=self.n_permutations,
            dtype=np.uint64,
        )
        self._b = generator.randint(
            low=0,
            high=self.prime,
            size=self.n_permutations,
            dtype=np.uint64,
        )

    # region Defaults
    @property
    def default_base_state(self):  # -> npt.NDArray[np.uint64]:
        return np.array([(2 ** 32) - 1] * self.n_permutations, dtype=np.uint64)

    @property
    def default_bound(self) -> np.uint64:
        return np.array((2 ** 32) - 1, dtype=np.uint64)

    @property
    def default_prime(self) -> np.uint64:
        return np.array((2 ** 61) - 1, dtype=np.uint64)

    # endregion

    # region Properties
    @property
    def a(self):  # -> npt.NDArray[np.uint64]:
        return self._a

    @property
    def b(self):  # -> npt.NDArray[np.uint64]:
        return self._b

    @property
    def base_state(self):  # -> npt.NDArray[np.uint64]:
        return np.copy(self._base_state)

    @property
    def bound(self):  # -> npt.NDArray[np.uint64]:
        return self._bound

    @property
    def n_permutations(self) -> int:
        return self._n_permutations

    @property
    def preprocess_function(self):  # -> Callable[[Any], npt.NDArray[np.uint64]]:
        return self._preprocess_function

    @property
    def prime(self) -> np.uint64:
        return self._prime

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def state(self):  # -> npt.NDArray[np.uint64]:
        return np.copy(self._state)

    # endregion

    # region Single implementation
    @staticmethod
    def preprocess(
        data: Any, preprocess_function  #: Callable[[Any], npt.NDArray[np.uint64]]
    ) -> Any:
        return preprocess_function(data)

    @staticmethod
    def data_to_base_hash(
        data,  #: npt.NDArray[np.uint64],
        a,  #: npt.NDArray[np.uint64],
        b,  #: npt.NDArray[np.uint64],
        prime: Any,
    ) -> np.uint64:
        return ((data * a) + b) % prime

    @staticmethod
    def base_hash_to_bounded_hash(
        base_hash,  #: npt.NDArray[np.uint64],
        bound: np.uint64,
    ):  # -> npt.NDArray[np.uint64]:
        return np.bitwise_and(base_hash, bound)

    @staticmethod
    def bounded_hash_to_minhash(
        bounded_hash,  #: npt.NDArray[np.uint64],
        state,  #: npt.NDArray[np.uint64],
    ):  # -> npt.NDArray[np.uint64]:
        return np.minimum(bounded_hash, state)

    # endregion

    # region Batch implementation
    @staticmethod
    def preprocess_batch(
        data_batch: Iterable[Any], preprocess_function: Callable[[Any], Any]
    ) -> Iterable[Any]:
        for data in data_batch:
            yield NumPyMinHash.preprocess(
                data=data, preprocess_function=preprocess_function
            )

    @staticmethod
    def data_batch_to_base_hashes(
        data_batch,  #: Iterable[npt.NDArray[np.uint64]],
        a,  #: npt.NDArray[np.uint64],
        b,  #: npt.NDArray[np.uint64],
        prime: np.uint64,
    ):  # -> Iterable[npt.NDArray[np.uint64]]:
        for data in data_batch:
            yield NumPyMinHash.data_to_base_hash(data=data, a=a, b=b, prime=prime)

    @staticmethod
    def base_hashes_to_bounded_hashes(
        base_hashes,  #: Iterable[npt.NDArray[np.uint64]],
        bound: np.uint64,
    ):  # -> Iterable[npt.NDArray[np.uint64]]:
        for base_hash in base_hashes:
            yield NumPyMinHash.base_hash_to_bounded_hash(
                base_hash=base_hash, bound=bound
            )

    @staticmethod
    def bounded_hashes_to_minhash(
        bounded_hashes,  #: Iterable[npt.NDArray[np.uint64]],
        state,  #: npt.NDArray[np.uint64],
    ):  # -> npt.NDArray[np.uint64]:
        return np.minimum(
            np.array(list(bounded_hashes)).min(axis=0),
            state,
        )

    # endregion
