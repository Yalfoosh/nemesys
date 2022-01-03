from typing import Tuple, Union

from nemesys.modelling.blocks.block import Block

class ShapedBlock(Block):
    def __init__(self, base_shape: Union[int, Tuple[int, ...]]): ...
    @property
    def base_shape(self) -> Tuple[int, ...]: ...
