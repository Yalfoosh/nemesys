import torch


class DeviceConversion:
    @staticmethod
    def to_torch(device):
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
    def to_torch(dtype):
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
    def to_tuple(shape):
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


class IterableConversion:
    @staticmethod
    def to_readable_string(
        iterable,
        default_string=None,
        separator=None,
        last_prefix=None,
    ):
        if default_string is None:
            default_string = ""
        if separator is None:
            separator = ", "
        if last_prefix is None:
            last_prefix = separator

        iterable = list(sorted([str(x) for x in iterable]))

        if len(iterable) == 0:
            return default_string
        elif len(iterable) == 1:
            return iterable[0]
        else:
            return separator.join(iterable[:-1]) + f"{last_prefix}{iterable[-1]}"
