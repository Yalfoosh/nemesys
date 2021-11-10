from typing import Iterable, List

import numpy as np

from nemesys.modelling.blocks.pytorch_block import PyTorchBlock
from nemesys.modelling.stores.list_store import ListStore
from nemesys.modelling.stores.pytorch_store import PyTorchStore


class PyTorchListStore(PyTorchStore, ListStore):
    def __init__(self):
        super().__init__()

    # region Store implementation
    """
    @staticmethod
    def init_from(content: Any, method: Optional[str]) -> "Store":
        raise NotImplementedError
    """

    @property
    def blocks(self) -> List[PyTorchBlock]:
        return self._blocks

    def append(self, content: PyTorchBlock):
        self._blocks.append(content.clone())

    def get_all(self) -> Iterable[PyTorchBlock]:
        return self.get_some(keys=range(len(self._blocks)))

    def get_one(self, key: int) -> PyTorchBlock:
        return self._blocks[key]

    def get_some(self, keys: Iterable[int]) -> Iterable[PyTorchBlock]:
        for key in keys:
            yield self.get_one(key=key)

    def remove_all(self):
        self._blocks.clear()

    def remove_one(self, key: int):
        self._blocks.pop(key)

    def remove_some(self, keys: Iterable[int]):
        for key in sorted(keys, reverse=True):
            self.remove_one(key=key)

    def set_all(self, content: Iterable[PyTorchBlock]):
        self._blocks = list(content)

    def set_one(self, key: int, content: PyTorchBlock):
        self._blocks[key] = content

    def set_some(self, keys: Iterable[int], contents: Iterable[PyTorchBlock]):
        for key, content in zip(keys, contents):
            self.set_one(key=key, content=content)

    # endregion

    # region Dunder methods
    def __str__(self):
        return str(np.array([block.data.cpu().numpy() for block in self.blocks]))

    # endregion
