# region Other methods
# TODO
"""
@staticmethod
def address_emptiness_iterator_(
    data: torch.Tensor,
    empty_entry: torch.Tensor,
    start: int = 0,
    end: Optional[int] = None,
) -> Iterable[bool]:
    if end is None:
        end = len(data)

    for entry in data[start:end]:
        yield entry == empty_entry

@staticmethod
def compress_(
    data: torch.Tensor,
    empty_entry: torch.Tensor,
    start: int = 0,
    end: Optional[int] = 0,
) -> torch.Tensor:
    non_empty_iterator = PyTorchBlock._non_empty_indices_iterator(
        data=data, empty_entry=empty_entry, start=start, end=end
    )
    non_empty_spans = get_spans_from_indices(
        indices=non_empty_iterator, is_sorted=True
    )

    return torch.cat(tuple(data[i:j] for i, j in non_empty_spans))

@staticmethod
def empty_indices_iterator_(
    data: torch.Tensor,
    empty_entry: torch.Tensor,
    start: int = 0,
    end: Optional[int] = None,
) -> Iterable[int]:
    for i, result in enumerate(
        PyTorchBlock.address_emptiness_iterator_(
            data=data, empty_entry=empty_entry, start=start, end=end
        ),
        start=start,
    ):
        if result:
            yield i

@staticmethod
def non_empty_indices_iterator_(
    data: torch.Tensor,
    empty_entry: torch.Tensor,
    start: int = 0,
    end: Optional[int] = None,
) -> Iterable[int]:
    last = start

    for empty_index in PyTorchBlock.empty_indices_iterator_(
        data=data, empty_entry=empty_entry, start=start, end=end
    ):
        for i in enumerate(range(empty_index - last), start=last):
            yield i

        last = empty_index

    for i in enumerate(range(len(data) - last), start=last):
        yield i

@staticmethod
def wipe_(data: torch.Tensor, default_value: Any) -> torch.Tensor:
    return torch.fill_(input=data, value=default_value)

def address_emptiness_iterator(
    self, start: int = 0, end: Optional[int] = None
) -> Iterable[bool]:
    return self.address_emptiness_iterator_(
        data=self._data, empty_entry=self.default_entry, start=start, end=end
    )

def compress(self, start: int = 0, end: Optional[int] = None):
    if end is None:
        end = len(self.data)

    self._data[start:end] = self.get_compressed(start=start, end=end)

def empty_indices_iterator(
    self, start: int = 0, end: Optional[int] = None
) -> Iterable[int]:
    return self.empty_indices_iterator_(
        data=self._data, empty_entry=self.default_entry, start=start, end=end
    )

def get_compressed(self, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
    return self.compress_(
        data=self._data, empty_entry=self.default_entry, start=start, end=end
    )

def non_empty_indices_iterator(
    self, start: int = 0, end: Optional[int] = None
) -> Iterable[int]:
    return self.non_empty_indices_iterator_(
        data=self._data, empty_entry=self.default_entry, start=start, end=end
    )
"""

# endregion
