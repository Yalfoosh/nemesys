from typing import Any, Callable, Iterable, Optional


class MinHash:
    def __init__(
        self,
        n_permutations: Any,
        seed: Any,
        preprocess_function: Callable[[Any], Any],
        prime: Optional[Any] = None,
        base_state: Optional[Any] = None,
        bound: Optional[Any] = None,
    ):
        self._n_permutations = n_permutations
        self._seed = seed
        self._preprocess_function = preprocess_function

        self._prime = prime
        self._base_state = base_state
        self._state = None
        self._bound = bound

        self.clear()

    # region Properties
    @property
    def a(self) -> Any:
        raise NotImplementedError

    @property
    def b(self) -> Any:
        raise NotImplementedError

    @property
    def base_state(self) -> Any:
        return self._base_state

    @property
    def bound(self) -> Any:
        return self._bound

    @property
    def n_permutations(self) -> Any:
        return self._n_permutations

    @property
    def preprocess_function(self) -> Callable[[Any], Any]:
        return self._preprocess_function

    @property
    def prime(self) -> Any:
        return self._prime

    @property
    def seed(self) -> Any:
        return self._seed

    @property
    def state(self) -> Any:
        return self._state

    @state.getter
    def state(self, value: Any):
        self._state = value

    # endregion

    # region Single implementation
    @staticmethod
    def preprocess(data: Any, preprocess_function: Callable[[Any], Any]) -> Any:
        raise NotImplementedError

    @staticmethod
    def data_to_base_hash(data: Any, a: Any, b: Any, prime: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def base_hash_to_bounded_hash(base_hash: Any, bound: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def bounded_hash_to_minhash(bounded_hash: Any, state: Any) -> Any:
        raise NotImplementedError

    # endregion

    # region Batch implementation
    @staticmethod
    def preprocess_batch(
        data_batch: Iterable[Any], preprocess_function: Callable[[Any], Any]
    ) -> Iterable[Any]:
        raise NotImplementedError

    @staticmethod
    def data_batch_to_base_hashes(
        data_batch: Iterable[Any], a: Any, b: Any, prime: Any
    ) -> Iterable[Any]:
        raise NotImplementedError

    @staticmethod
    def base_hashes_to_bounded_hashes(
        base_hashes: Iterable[Any], bound: Any
    ) -> Iterable[Any]:
        raise NotImplementedError

    @staticmethod
    def bounded_hashes_to_minhash(bounded_hashes: Iterable[Any], state: Any) -> Any:
        raise NotImplementedError

    # endregion

    # region Single hashing
    def get_hash_eager(self, data: Any) -> Any:
        preprocessed = self.preprocess(
            data=data, preprocess_function=self.preprocess_function
        )
        base_hash = self.data_to_base_hash(
            data=preprocessed, a=self.a, b=self.b, prime=self.prime
        )
        bounded_hash = self.base_hash_to_bounded_hash(
            base_hash=base_hash, bound=self.bound
        )

        return bounded_hash

    def get_minhash_eager(self, data: Any, state: Any) -> Any:
        bounded_hash = self.get_hash_eager(data=data)
        minhash = self.bounded_hash_to_minhash(
            bounded_hash=bounded_hash, hash_state=state
        )

        return minhash

    def update(self, data: Any) -> Any:
        self.state = self.get_minhash_eager(data=data, state=self.state)

    # endregion

    # region Batch hashing
    def get_hash_batch_eager(self, data_batch: Iterable[Any]) -> Iterable[Any]:
        preprocessed_batch = self.preprocess_batch(
            data_batch=data_batch, preprocess_function=self.preprocess_function
        )
        base_hashes = self.data_batch_to_base_hashes(
            data_batch=preprocessed_batch, a=self.a, b=self.b, prime=self.prime
        )
        bounded_hashes = self.base_hashes_to_bounded_hashes(
            base_hashes=base_hashes, bound=self.bound
        )

        return bounded_hashes

    def get_minhash_batch_eager(self, data_batch: Iterable[Any], state: Any) -> Any:
        bounded_hashes = self.get_hash_batch_eager(data_batch=data_batch)
        minhash = self.bounded_hashes_to_minhash(
            bounded_hashes=bounded_hashes, state=state
        )

        return minhash

    def update_batch(self, data_batch: Iterable[Any]) -> Any:
        self.state = self.get_minhash_batch_eager(
            data_batch=data_batch, state=self.state
        )

    # endregion

    # region Multiple hashing
    def get_hash_many_eager(self, data_many: Iterable[Any]) -> Iterable[Any]:
        for data in data_many:
            yield self.get_hash_eager(data=data)

    def get_minhash_many_eager(
        self, data_many: Iterable[Any], state: Any
    ) -> Iterable[Any]:
        for data in data_many:
            yield self.get_minhash_eager(data=data, state=state)

    # endregion

    # region Multiple batch hashing
    def get_hash_batch_many_eager(
        self, data_batch_many: Iterable[Iterable[Any]]
    ) -> Iterable[Any]:
        for data_batch in data_batch_many:
            yield self.get_hash_batch_eager(data_batch=data_batch)

    def get_minhash_batch_many_eager(
        self, data_batch_many: Iterable[Iterable[Any]], state: Any
    ) -> Iterable[Any]:
        for data_batch in data_batch_many:
            yield self.get_minhash_batch_eager(data_batch=data_batch, state=state)

    # endregion

    def clear(self):
        self.state = self.base_state

    def digest(self) -> Any:
        return self.state
