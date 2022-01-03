from typing import Any, Iterable

from nemesys.modelling.routers.router import Router

class ConcatenationRouter(Router):
    def concatenate(self, inputs: Iterable[Any]) -> Any: ...
    def route(self, inputs: Iterable[Any]) -> Any: ...
