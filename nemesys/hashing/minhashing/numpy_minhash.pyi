from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import numpy.typing as npt

from nemesys.hashing.minhashing.minhash import MinHash

HashValueType = np.uint64
HashType = npt.NDArray[HashValueType]
PreprocessFunctionType = Callable[[Any], HashType]

class NumPyMinHash(MinHash):
    def __init__(
        self,
        n_permutations: int,
        seed: int,
        preprocess_function: PreprocessFunctionType,
        prime: Optional[Union[HashValueType, int]],
        base_state: Optional[Iterable[Union[np.uint32, int]]],
        bound: Optional[Union[HashValueType, int]],
    ): ...
    @property
    def default_base_state(self) -> HashType: ...
    @property
    def default_bound(self) -> HashValueType: ...
    @property
    def default_prime(self) -> HashValueType: ...
    @property
    def a(self) -> HashType: ...
    @property
    def b(self) -> HashType: ...
    @property
    def base_state(self) -> HashType: ...
    @property
    def bound(self) -> HashType: ...
    @property
    def n_permutations(self) -> int: ...
    @property
    def preprocess_function(self) -> PreprocessFunctionType: ...
    @property
    def prime(self) -> HashValueType: ...
    @property
    def seed(self) -> int: ...
    @property
    def state(self) -> HashType: ...
    @staticmethod
    def preprocess(
        data: Any, preprocess_function: PreprocessFunctionType
    ) -> HashType: ...
    @staticmethod
    def data_to_base_hash(
        data: HashType, a: HashType, b: HashType, prime: Any
    ) -> HashType: ...
    @staticmethod
    def base_hash_to_bounded_hash(
        base_hash: HashType, bound: HashValueType
    ) -> HashType: ...
    @staticmethod
    def bounded_hash_to_minhash(
        bounded_hash: HashType, state: HashType
    ) -> HashType: ...
    @staticmethod
    def preprocess_batch(
        data_batch: Iterable[Any], preprocess_function: PreprocessFunctionType
    ) -> Iterable[HashType]: ...
    @staticmethod
    def data_batch_to_base_hashes(
        data_batch: HashType, a: HashType, b: HashType, prime: HashValueType
    ) -> Iterable[HashType]: ...
    @staticmethod
    def base_hashes_to_bounded_hashes(
        base_hashes: Iterable[HashType], bound: HashValueType
    ) -> Iterable[HashType]: ...
    @staticmethod
    def bounded_hashes_to_minhash(
        bounded_hashes: HashType, state: HashType
    ) -> HashType: ...
