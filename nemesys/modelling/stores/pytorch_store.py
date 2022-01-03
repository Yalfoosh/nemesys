from nemesys.modelling.stores.store import Store


class PyTorchStore(Store):
    @property
    def blocks(self):
        raise NotImplementedError

    @staticmethod
    def init_from(content, method):
        raise NotImplementedError

    def append(self, content):
        raise NotImplementedError

    def get_all(self):
        raise NotImplementedError

    def get_one(self, key):
        raise NotImplementedError

    def get_some(self, keys):
        raise NotImplementedError

    def remove_all(self):
        raise NotImplementedError

    def remove_one(self, key):
        raise NotImplementedError

    def remove_some(self, keys):
        raise NotImplementedError

    def set_all(self, content):
        raise NotImplementedError

    def set_one(self, key, content):
        raise NotImplementedError

    def set_some(self, keys, contents):
        raise NotImplementedError
