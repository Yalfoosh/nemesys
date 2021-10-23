from typing import Iterable

from nemesys.modelling.blocks.block import Block


class Store:
    @property
    def blocks(self) -> Iterable[Block]:
        raise NotImplementedError
