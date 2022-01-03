from typing import Any, Iterable, Optional, Union, Tuple

import torch

class DeviceConversion:
    @staticmethod
    def to_torch(device: Union[str, torch.device]) -> torch.device: ...

class DtypeConversion:
    @staticmethod
    def to_torch(dtype: Union[str, torch.dtype]) -> torch.dtype: ...

class ShapeConversion:
    @staticmethod
    def to_tuple(shape: Union[int, Iterable[int]]) -> Tuple[int, ...]: ...

class IterableConversion:
    @staticmethod
    def to_readable_string(
        iterable: Iterable[Any],
        default_string: Optional[str],
        separator: Optional[str],
        last_prefix: Optional[str],
    ) -> str: ...
