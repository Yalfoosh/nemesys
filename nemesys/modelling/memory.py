from typing import Any

import torch

from nemesys.modelling.block import Block
from nemesys.utils.exceptions import SizeException
from nemesys.utils.preparation import TensorPreparation
from nemesys.utils.spans import get_spans_from_indices


class BasicMemory:
    def __init__(self, block: Block, capacity: int = 0):
        self._block = block

        self.initialize_block(size=capacity)

    # region Factories

    @staticmethod
    def from_tensor(
        tensor: torch.Tensor,
        default_value: Any,
        allow_padding: bool = False,
        allow_reshaping: bool = False,
    ) -> "BasicMemory":
        base_shape = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[1:]

        block = Block(
            base_shape=base_shape,
            dtype=tensor.dtype,
            device=tensor.device,
            default_value=default_value,
        )

        memory = BasicMemory(block=block, capacity=0)
        memory.append(
            tensor, allow_padding=allow_padding, allow_reshaping=allow_reshaping
        )

        return memory

    # endregion

    def append(
        self,
        data: torch.Tensor,
        allow_padding: bool = False,
        allow_reshaping: bool = False,
    ):
        data = TensorPreparation.for_block_insertion(
            tensor=data,
            dtype=self._block.dtype,
            device=self._block.device,
            base_shape=self._block.base_shape,
            allow_padding=allow_padding,
            allow_reshaping=allow_reshaping,
        )

        self._block.content = torch.cat((self._block.content, data))

    def clear(self):
        Block.deallocate(storage=self._block.content)
        self.initialize_block(size=0)

    def compress(self):
        non_empty_indices = self.get_non_empty_indices()
        non_empty_spans = get_spans_from_indices(
            indices=non_empty_indices, is_sorted=True
        )

        self._block.content = torch.cat(
            [self._block.content[begin:end] for begin, end in non_empty_spans]
        )

    def crop(self, start: int, end: int):
        self._block.content = self._block.content[start:end]

    def get_empty_entry(self):
        return self.reserve(size=1)[0]

    def get_empty_indices(self):
        empty_entry = self.get_empty_entry()

        empty_indices = [
            i
            for i, entry in enumerate(self._block.content)
            if torch.equal(entry, empty_entry)
        ]

        del empty_entry

        return empty_indices

    def get_non_empty_indices(self):
        empty_entry = self.get_empty_entry()
        non_empty_indices = [
            i
            for i, entry in enumerate(self._block.content)
            if not torch.equal(entry, empty_entry)
        ]

        del empty_entry

        return non_empty_indices

    def initialize_block(self, size: int):
        self._block.content = Block.allocate(
            size=size,
            shape=self._block.base_shape,
            dtype=self._block.dtype,
            device=self._block.device,
            fill_value=self._block.default_value,
        )

    def reserve(self, size: int = 1) -> torch.Tensor:
        try:
            size = int(size)
        except TypeError:
            raise SizeException("Size couldn't be cast to int")

        if size < 0:
            raise SizeException(f"Size must be 0 or greater, but it is {size}")

        return Block.allocate(
            size=size,
            shape=self._block.base_shape,
            dtype=self._block.dtype,
            device=self._block.device,
            fill_value=self._block.default_value,
        )

    def wipe(self):
        torch.fill_(self._block.content, self._block.default_value)

    def __getitem__(self, key):
        return self._block.content.__getitem__(key)

    def __setitem__(self, key, value):
        return self._block.content.__setitem__(key, value)

    def __len__(self):
        return len(self._block.content)

    def __repr__(self):
        return repr(self._block.content)

    def __str__(self):
        return str(self._block.content)
