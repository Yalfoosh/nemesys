class MinHash:
    def __init__(
        self,
        n_permutations,
        seed,
        preprocess_function,
        prime=None,
        base_state=None,
        bound=None,
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
    def a(self):
        raise NotImplementedError

    @property
    def b(self):
        raise NotImplementedError

    @property
    def base_state(self):
        return self._base_state

    @property
    def bound(self):
        return self._bound

    @property
    def n_permutations(self):
        return self._n_permutations

    @property
    def preprocess_function(self):
        return self._preprocess_function

    @property
    def prime(self):
        return self._prime

    @property
    def seed(self):
        return self._seed

    @property
    def state(self):
        return self._state

    @state.getter
    def state(self, value):
        self._state = value

    # endregion

    # region Single implementation
    @staticmethod
    def preprocess(data, preprocess_function):
        raise NotImplementedError

    @staticmethod
    def data_to_base_hash(data, a, b, prime):
        raise NotImplementedError

    @staticmethod
    def base_hash_to_bounded_hash(base_hash, bound):
        raise NotImplementedError

    @staticmethod
    def bounded_hash_to_minhash(bounded_hash, state):
        raise NotImplementedError

    # endregion

    # region Batch implementation
    @staticmethod
    def preprocess_batch(data_batch, preprocess_function):
        raise NotImplementedError

    @staticmethod
    def data_batch_to_base_hashes(data_batch, a, b, prime):
        raise NotImplementedError

    @staticmethod
    def base_hashes_to_bounded_hashes(base_hashes, bound):
        raise NotImplementedError

    @staticmethod
    def bounded_hashes_to_minhash(bounded_hashes, state):
        raise NotImplementedError

    # endregion

    # region Single hashing
    def get_hash_eager(self, data):
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

    def get_minhash_eager(self, data, state):
        if state is None:
            state = self.base_state

        bounded_hash = self.get_hash_eager(data=data)
        minhash = self.bounded_hash_to_minhash(
            bounded_hash=bounded_hash, hash_state=state
        )

        return minhash

    def update(self, data):
        self._state = self.get_minhash_eager(data=data, state=self.state)

    # endregion

    # region Batch hashing
    def get_hash_batch_eager(self, data_batch):
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

    def get_minhash_batch_eager(self, data_batch, state=None):
        if state is None:
            state = self.base_state

        bounded_hashes = self.get_hash_batch_eager(data_batch=data_batch)
        minhash = self.bounded_hashes_to_minhash(
            bounded_hashes=bounded_hashes, state=state
        )

        return minhash

    def update_batch(self, data_batch):
        self._state = self.get_minhash_batch_eager(
            data_batch=data_batch, state=self.state
        )

    # endregion

    # region Multiple hashing
    def get_hash_many_eager(self, data_many):
        for data in data_many:
            yield self.get_hash_eager(data=data)

    def get_minhash_many_eager(self, data_many, state=None):
        if state is None:
            state = self.base_state

        for data in data_many:
            yield self.get_minhash_eager(data=data, state=state)

    # endregion

    # region Multiple batch hashing
    def get_hash_batch_many_eager(self, data_batch_many):
        for data_batch in data_batch_many:
            yield self.get_hash_batch_eager(data_batch=data_batch)

    def get_minhash_batch_many_eager(self, data_batch_many, state=None):
        if state is None:
            state = self.base_state

        for data_batch in data_batch_many:
            yield self.get_minhash_batch_eager(data_batch=data_batch, state=state)

    # endregion

    def clear(self):
        self._state = self.base_state

    def digest(self):
        return self.state
