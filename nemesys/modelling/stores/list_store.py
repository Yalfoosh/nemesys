import pprint
from typing import Iterable, List

from nemesys.modelling.blocks.block import Block
from nemesys.modelling.stores.store import Store
from nemesys.utils.re import WHITESPACE_RE


class ListStore(Store):
    def __init__(self):
        super().__init__()

        self._blocks = list()

    # region Store implementation
    @property
    def blocks(self) -> List[Block]:
        return self._blocks

    def append(self, content: Block):
        self._blocks.append(content.clone())

    def get_all(self) -> Iterable[Block]:
        return self.get_some(keys=range(len(self._blocks)))

    def get_one(self, key: int) -> Block:
        return self._blocks[key]

    def get_some(self, keys: Iterable[int]) -> Iterable[Block]:
        for key in keys:
            yield self.get_one(key=key)

    def remove_all(self):
        self._blocks.clear()

    def remove_one(self, key: int):
        self._blocks.pop(key)

    def remove_some(self, keys: Iterable[int]):
        for key in sorted(keys, reverse=True):
            self.remove_one(key=key)

    def set_all(self, content: Iterable[Block]):
        self._blocks = content

    def set_one(self, key: int, content: Block):
        self._blocks[key] = content

    def set_some(self, keys: Iterable[int], contents: Iterable[Block]):
        for key, content in zip(keys, contents):
            self._blocks[key] = content

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
