from typing import Any, Iterable, Optional

from nemesys.modelling.stores.store import Store
from nemesys.modelling.blocks.pytorch_block import PyTorchBlock


class PyTorchStore(Store):
    @property
    def blocks(self) -> Iterable[PyTorchBlock]:
        raise NotImplementedError

    @staticmethod
    def init_from(content: Any, method: Optional[str]) -> "PyTorchStore":
        raise NotImplementedError

    def append(self, content: PyTorchBlock):
        raise NotImplementedError

    def get_all(self) -> Iterable[PyTorchBlock]:
        raise NotImplementedError

    def get_one(self, key: Any) -> PyTorchBlock:
        raise NotImplementedError

    def get_some(self, keys: Iterable[Any]) -> Iterable[PyTorchBlock]:
        raise NotImplementedError

    def remove_all(self):
        raise NotImplementedError

    def remove_one(self, key: Any):
        raise NotImplementedError

    def remove_some(self, keys: Iterable[Any]):
        raise NotImplementedError

    def set_all(self, content: Iterable[PyTorchBlock]):
        raise NotImplementedError

    def set_one(self, key: Any, content: PyTorchBlock):
        raise NotImplementedError

    def set_some(self, keys: Iterable[Any], contents: Iterable[PyTorchBlock]):
        raise NotImplementedError
