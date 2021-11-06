from typing import Any, Iterable


class Router:
    def route(self, inputs: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError
