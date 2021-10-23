import copy
from typing import Any, Iterable, Optional, Tuple, Union

from nemesys.utils.conversion import ShapeConversion


class Block:
    def __init__(self, size: int = 0):
        self.allocate(size=size)

    @property
    def data(self):
        raise NotImplementedError

    def allocate(self, size: int):
        raise NotImplementedError

    def deallocate(self, size: Optional[int]):
        raise NotImplementedError

    def reallocate(self, new_size: int):
        raise NotImplementedError

    def __del__(self):
        self.deallocate(size=None)


class ShapedBlock(Block):
    def __init__(self, base_shape: Union[int, Tuple[int, ...]], default_value: Any):
        self._base_shape = ShapeConversion.to_tuple(base_shape)
        self._default_value = copy.deepcopy(default_value)

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self._base_shape

    @property
    def default_value(self) -> Any:
        return self._default_value
