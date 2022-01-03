from typing import Any, Optional


class Block:
    @property
    def data(self):
        raise NotImplementedError

    @data.setter
    def data(self, value: Any):
        raise NotImplementedError

    @staticmethod
    def init_from(content: Any, method: Optional[str]) -> "Block":
        raise NotImplementedError

    def clone(self) -> "Block":
        return self.init_from(content=self.data)

    def default(self):
        raise NotImplementedError
