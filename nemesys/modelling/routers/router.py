from typing import Any, Iterable


class Router:
    def route(self, inputs: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError

    def __call__(self, inputs: Iterable[Any]) -> Iterable[Any]:
        return self.route(inputs=inputs)
