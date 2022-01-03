import pprint

from nemesys.modelling.stores.store import Store
from nemesys.utils.re import WHITESPACE_RE


class ListStore(Store):
    def __init__(self):
        super().__init__()

        self._blocks = list()

    # region Store implementation
    @property
    def blocks(self):
        return self._blocks

    def append(self, content):
        self._blocks.append(content.clone())

    def get_all(self):
        return self.get_some(keys=range(len(self._blocks)))

    def get_one(self, key):
        return self._blocks[key]

    def get_some(self, keys):
        for key in keys:
            yield self.get_one(key=key)

    def remove_all(self):
        self._blocks.clear()

    def remove_one(self, key):
        self._blocks.pop(key)

    def remove_some(self, keys):
        for key in sorted(keys, reverse=True):
            self.remove_one(key=key)

    def set_all(self, content):
        self._blocks = list(content)

    def set_one(self, key, content):
        self._blocks[key] = content

    def set_some(self, keys, contents):
        for key, content in zip(keys, contents):
            self.set_one(key=key, content=content)

    # endregion

    # region Dunder methods
    def __len__(self):
        return len(self._blocks)

    def __repr__(self):
        fixed_str = WHITESPACE_RE.sub(" ", str(self).strip())

        return f"{self.__class__.__name__}({len(self)} blocks) {fixed_str}"

    def __str__(self):
        return pprint.pformat(self._blocks, indent=2)

    # endregion
