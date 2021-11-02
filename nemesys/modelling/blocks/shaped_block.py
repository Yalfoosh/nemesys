from typing import Tuple, Union

from nemesys.modelling.blocks.block import Block
from nemesys.utils.conversion import ShapeConversion


class ShapedBlock(Block):
    def __init__(self, base_shape: Union[int, Tuple[int, ...]]):
        super().__init__()

        self._base_shape = ShapeConversion.to_tuple(base_shape)

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self._base_shape
