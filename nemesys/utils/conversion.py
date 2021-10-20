from typing import Iterable, Union

import torch


class DeviceConversion:
    @staticmethod
    def to_torch(device: Union[str, torch.device]):
        if isinstance(device, torch.device):
            return device
        else:
            try:
                device = str(device)
            except TypeError:
                raise TypeError("Couldn't cast device to str")

            try:
                device = torch.device(device)
            except Exception:
                raise ValueError(
                    f"Device `{device}` must be a valid {torch.device.__name__}"
                )

            return device


class DtypeConversion:
    @staticmethod
    def to_torch(dtype: Union[str, torch.dtype]):
        if isinstance(dtype, torch.dtype):
            return dtype
        else:
            try:
                dtype = str(dtype)
            except TypeError:
                raise TypeError("Couldn't cast dtype to str")

            try:
                dtype = getattr(torch, dtype)
            except AttributeError:
                raise ValueError(
                    f"Dtype `{dtype}` not found in {torch.__name__} module"
                )

            return dtype


class ShapeConversion:
    @staticmethod
    def to_tuple(shape: Union[int, Iterable[int]]):
        try:
            shape_tuple = (int(shape),)
        except TypeError:
            shape_tuple = tuple(int(x) for x in shape)

        for i, x in enumerate(shape_tuple):
            if x < 1:
                raise ValueError(
                    f"All shape tuple elements must be positive, but shape_tuple[{i}] is "
                    f"{x}"
                )

        return shape_tuple
