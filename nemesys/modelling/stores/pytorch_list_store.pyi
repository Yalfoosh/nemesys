from typing import Iterable, List

from nemesys.modelling.blocks.pytorch_block import PyTorchBlock
from nemesys.modelling.stores.list_store import ListStore
from nemesys.modelling.stores.pytorch_store import PyTorchStore

class PyTorchListStore(PyTorchStore, ListStore):
    def __init__(self):
        super().__init__()
    @property
    def blocks(self) -> List[PyTorchBlock]: ...
    def append(self, content: PyTorchBlock): ...
    def get_all(self) -> Iterable[PyTorchBlock]: ...
    def get_one(self, key: int) -> PyTorchBlock: ...
    def get_some(self, keys: Iterable[int]) -> Iterable[PyTorchBlock]: ...
    def remove_all(self): ...
    def remove_one(self, key: int): ...
    def remove_some(self, keys: Iterable[int]): ...
    def set_all(self, content: Iterable[PyTorchBlock]): ...
    def set_one(self, key: int, content: PyTorchBlock): ...
    def set_some(self, keys: Iterable[int], contents: Iterable[PyTorchBlock]): ...
    def __str__(self) -> str: ...