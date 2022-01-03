from typing import Any, Optional

class Block:
    @property
    def data(self): ...
    @data.setter
    def data(self, value: Any): ...
    @staticmethod
    def init_from(content: Any, method: Optional[str]) -> "Block": ...
    def clone(self) -> "Block": ...
    def default(self): ...