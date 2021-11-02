import copy
from typing import Any, Iterable, Optional, Tuple, Union

from nemesys.utils.conversion import ShapeConversion


class Block:
    @property
    def data(self) -> Any:
        raise NotImplementedError

    @staticmethod
    def init_from(content: Any, method: Optional[str]) -> "Block":
        raise NotImplementedError

    def default(self):
        raise NotImplementedError

    def read(self, key: Optional[Any]) -> Any:
        raise NotImplementedError

    def write(self, content: Optional[Any]):
        raise NotImplementedError


class ShapedBlock(Block):
    def __init__(self, base_shape: Union[int, Tuple[int, ...]]):
        super().__init__()

        self._base_shape = ShapeConversion.to_tuple(base_shape)

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self._base_shape
