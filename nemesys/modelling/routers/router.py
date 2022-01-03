class Router:
    def route(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.route(inputs=inputs)
