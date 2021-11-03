from typing import Any, Iterable, Optional

from nemesys.modelling.blocks.block import Block


class Store:
    @property
    def blocks(self) -> Iterable[Block]:
        raise NotImplementedError

    @staticmethod
    def init_from(content: Any, method: Optional[str]) -> "Store":
        raise NotImplementedError

    def append(self, content: Block):
        raise NotImplementedError

    def get_all(self) -> Iterable[Block]:
        raise NotImplementedError

    def get_one(self, key: Any) -> Block:
        raise NotImplementedError

    def get_some(self, keys: Iterable[Any]) -> Iterable[Block]:
        raise NotImplementedError

    def remove_all(self):
        raise NotImplementedError

    def remove_one(self, key: Any):
        raise NotImplementedError

    def remove_some(self, keys: Iterable[Any]):
        raise NotImplementedError

    def set_all(self, content: Iterable[Block]):
        raise NotImplementedError

    def set_one(self, key: Any, content: Block):
        raise NotImplementedError

    def set_some(self, keys: Iterable[Any], contents: Iterable[Block]):
        raise NotImplementedError
