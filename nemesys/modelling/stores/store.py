from math import prod
import pprint
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from nemesys.modelling.blocks.block import Block
from nemesys.utils.re import WHITESPACE_RE


class Store:
    # region Properties
    @property
    def blocks(self) -> Iterable[Block]:
        raise NotImplementedError

    # endregion

    @staticmethod
    def init_from(content: any, method: Optional[str]) -> "Store":
        raise NotImplementedError

    def append(self, content: Optional[Block]):
        raise NotImplementedError

    def read(self, key: Optional[Any]) -> Union[Block, Iterable[Block]]:
        raise NotImplementedError

    def remove(self, key: Optional[Any]):
        raise NotImplementedError


class ListStore(Store):
    def __init__(self):
        super().__init__()

        self._blocks: List[Block] = list()

    # region Properties
    @property
    def blocks(self) -> List[Block]:
        return self._blocks

    # endregion

    # region Static methods
    @staticmethod
    def prepare_append(content: Optional[Block], default_block: Block) -> Block:
        if content is None:
            content = default_block

        return content

    @staticmethod
    def prepare_read(
        blocks: List[Block], key: Optional[int]
    ) -> Union[Block, Iterable[Block]]:
        if key is None:
            return blocks
        else:
            return blocks[int(key)]

    @staticmethod
    def prepare_remove(
        n_blocks: int,
        key: Optional[Union[int, Iterable[int]]],
    ) -> List[Block]:
        if key is None:
            return list(range(n_blocks))
        else:
            try:
                return [int(key)]
            except TypeError:
                return list(set(key))

    # endregion

    # region Generation methods
    def get_appended(self, content: Optional[Block]) -> Block:
        return self.prepare_append(content=content, default_block=self._default_block)

    def get_read(
        self, key: Optional[Union[int, Iterable[int]]]
    ) -> Union[Block, Iterable[Block]]:
        return self.prepare_read(blocks=self._blocks, key=key)

    def get_removed(self, key: Optional[Union[int, Iterable[int]]]) -> List[Block]:
        return self.prepare_remove(n_blocks=len(self._blocks), key=key)

    # endregion

    # region Store implementation
    def append(self, content: Optional[Block] = None):
        self._blocks.append(self.get_appended(content=content))

    def read(
        self, key: Optional[Union[int, Iterable[int]]]
    ) -> Union[Block, Iterable[Block]]:
        return self.get_read(key=key)

    def remove(self, key: Optional[Union[int, Iterable[int]]]):
        indices_to_remove = set(self.get_removed(key=key))

        for index in indices_to_remove:
            del self._blocks[index]

        self._blocks = [
            block
            for i, block in enumerate(indices_to_remove)
            if i not in indices_to_remove
        ]

    # endregion

    # region Dunder methods
    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self):
        fixed_str = WHITESPACE_RE.sub(" ", str(self).strip())

        return f"{self.__class__.__name__}({len(self)} blocks) {fixed_str}"

    def __str__(self):
        return pprint.pformat(self._blocks, indent=2)

    # endregion
