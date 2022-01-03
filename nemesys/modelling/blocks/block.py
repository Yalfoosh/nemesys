class Block:
    @property
    def data(self):
        raise NotImplementedError

    @data.setter
    def data(self, value):
        raise NotImplementedError

    @staticmethod
    def init_from(content, method):
        raise NotImplementedError

    def clone(self):
        return self.init_from(content=self.data)

    def default(self):
        raise NotImplementedError
