from typing import Any, Optional


class Block:
    @property
    def data(self):
        raise NotImplementedError

    @data.setter
    def data(self):
        raise NotImplementedError

    @staticmethod
    def init_from(content: Any, method: Optional[str]) -> "Block":
        raise NotImplementedError

    def default(self):
        raise NotImplementedError
