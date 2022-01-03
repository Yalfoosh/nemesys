from typing import Any, Iterable


from nemesys.modelling.routers.router import Router


class ConcatenationRouter(Router):
    def concatenate(self, inputs: Iterable[Any]) -> Any:
        raise NotImplementedError

    def route(self, inputs: Iterable[Any]) -> Any:
        return self.concatenate(inputs=inputs)
