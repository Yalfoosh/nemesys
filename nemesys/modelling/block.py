from typing import Any, Iterable, Optional, Union

import torch

from nemesys.utils.conversion import DeviceConversion, DtypeConversion, ShapeConversion


class Block:
    def __init__(
        self,
        base_shape: Union[int, Iterable[int]],
        dtype: Union[str, torch.dtype] = "float32",
        device: Union[str, torch.device] = "cpu",
        default_value: Any = 0.0,
    ):
        self._base_shape = ShapeConversion.to_tuple(base_shape)
        self._dtype = DtypeConversion.to_torch(dtype)
        self._device = DeviceConversion.to_torch(device)
        self._default_value = default_value

        self.content: Optional[torch.Tensor] = None

    # region Properties
    @property
    def base_shape(self) -> Iterable[int]:
        return self._base_shape

    @property
    def default_value(self) -> Any:
        return self._default_value

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> Iterable[int]:
        return self._content.shape

    # endregion

    # region Static methods
    @staticmethod
    def allocate(
        size: int,
        shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device,
        fill_value: Any,
    ):
        return torch.full(
            size=(size, *shape),
            fill_value=fill_value,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    @staticmethod
    def deallocate(storage: torch.Tensor):
        del storage

    @staticmethod
    def reallocate(storage: torch.Tensor, size: int, fill_value: Any):
        size_difference = size - storage.shape[0]

        if size_difference == 0:
            return storage
        elif size_difference < 0:
            return storage[:size_difference]
        else:
            return torch.cat(
                (
                    storage,
                    Block.allocate(
                        size=size_difference,
                        shape=storage.shape[1:],
                        dtype=storage.dtype,
                        device=storage.device,
                        fill_value=fill_value,
                    ),
                )
            )

    # endregion
