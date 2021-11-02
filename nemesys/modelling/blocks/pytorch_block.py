from typing import Any, Iterable, Tuple, Union, Optional

import torch

from nemesys.modelling.blocks.block import ShapedBlock
from nemesys.utils.conversion import DeviceConversion, DtypeConversion
from nemesys.utils.preparation import TensorPreparation
from nemesys.utils.re import WHITESPACE_RE
from nemesys.utils.spans import get_spans_from_indices


class PyTorchBlock(ShapedBlock):
    def __init__(
        self,
        base_shape: Union[int, Iterable[int]],
        dtype: Union[str, torch.dtype] = "float32",
        device: Union[str, torch.device] = "cpu",
        default_value: Any = 0.0,
    ):
        super().__init__(base_shape=base_shape, default_value=default_value)

        self._dtype = DtypeConversion.to_torch(dtype)
        self._device = DeviceConversion.to_torch(device)

        self._data = None

        self.allocate(size=0)

    # region Properties
    @property
    def data(self) -> Optional[torch.Tensor]:
        return self._data

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def default_entry(self) -> torch.Tensor:
        return self.get_allocated(size=1)

    # endregion

    # region Allocation
    def allocate_(
        self,
        size: int,
        base_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        default_value: Any = 0.0,
    ):
        return torch.full(
            size=(size, *base_shape),
            fill_value=default_value,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    def get_allocated(self, size: int) -> torch.Tensor:
        return self.allocate_(
            size=size,
            base_shape=self.base_shape,
            dtype=self.dtype,
            device=self.device,
            default_value=self.default_value,
        )

    def allocate(self, size: int):
        self._data = self.get_allocated(size=size)

    # ------------------------------------------------------------

    def deallocate_(
        self, data: torch.Tensor, size: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        if size is None or size >= len(self.data):
            del data
            return None
        else:
            return data[: len(data) - size]

    def get_deallocated(self, size: Optional[int] = None) -> Optional[torch.Tensor]:
        return self.deallocate_(data=self._data, size=size)

    def deallocate(self, size: Optional[int] = None):
        self._data = self.get_deallocated(size=size)

    # ------------------------------------------------------------

    def reallocate_(self, data: torch.Tensor, new_size: int) -> torch.Tensor:
        size_difference = len(data) - new_size

        if size_difference == 0:
            return data
        elif size_difference < 0:
            return self.deallocate(data=data, size=-size_difference)
        else:
            return torch.cat((data, self.allocate(size=size_difference)))

    def get_reallocated(self, new_size: int) -> Optional[torch.Tensor]:
        return self.reallocate_(data=self._data, new_size=new_size)

    def reallocate(self, new_size: int):
        self._data = self.get_reallocated(new_size=new_size)

    # endregion

    # region Static methods
    @staticmethod
    def address_emptiness_iterator_(
        data: torch.Tensor,
        empty_entry: torch.Tensor,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Iterable[bool]:
        if end is None:
            end = len(data)

        for entry in data[start:end]:
            yield entry == empty_entry

    @staticmethod
    def append_(
        data: torch.Tensor,
        new_data: torch.Tensor,
        allow_padding: bool = False,
        allow_reshaping: bool = False,
    ) -> torch.Tensor:
        new_data = TensorPreparation.for_block_insertion(
            tensor=new_data,
            dtype=data.dtype,
            device=data.device,
            base_shape=data.shape[1:],
            allow_padding=allow_padding,
            allow_reshaping=allow_reshaping,
        )

        return torch.cat((data, new_data))

    @staticmethod
    def clear_(block: "PyTorchBlock"):
        block.deallocate(size=None)
        block.allocate(size=0)

    @staticmethod
    def compress_(
        data: torch.Tensor,
        empty_entry: torch.Tensor,
        start: int = 0,
        end: Optional[int] = 0,
    ) -> torch.Tensor:
        non_empty_iterator = PyTorchBlock._non_empty_indices_iterator(
            data=data, empty_entry=empty_entry, start=start, end=end
        )
        non_empty_spans = get_spans_from_indices(
            indices=non_empty_iterator, is_sorted=True
        )

        return torch.cat(tuple(data[i:j] for i, j in non_empty_spans))

    @staticmethod
    def empty_indices_iterator_(
        data: torch.Tensor,
        empty_entry: torch.Tensor,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Iterable[int]:
        for i, result in enumerate(
            PyTorchBlock.address_emptiness_iterator_(
                data=data, empty_entry=empty_entry, start=start, end=end
            ),
            start=start,
        ):
            if result:
                yield i

    @staticmethod
    def from_tensor_(tensor: torch.Tensor, default_value: Any = 0.0) -> "PyTorchBlock":
        base_shape = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[1:]

        block = PyTorchBlock(
            base_shape=base_shape,
            dtype=tensor.dtype,
            device=tensor.device,
            default_value=default_value,
        )

        # No padding and reshaping is allowed since tensor should
        # fit in block perfectly.
        block.append(new_data=tensor, allow_padding=False, allow_reshaping=False)

        return block

    @staticmethod
    def non_empty_indices_iterator_(
        data: torch.Tensor,
        empty_entry: torch.Tensor,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Iterable[int]:
        last = start

        for empty_index in PyTorchBlock.empty_indices_iterator_(
            data=data, empty_entry=empty_entry, start=start, end=end
        ):
            for i in enumerate(range(empty_index - last), start=last):
                yield i

            last = empty_index

        for i in enumerate(range(len(data) - last), start=last):
            yield i

    @staticmethod
    def wipe_(data: torch.Tensor, default_value: Any) -> torch.Tensor:
        return torch.fill_(input=data, value=default_value)

    # endregion

    # region Other methods
    def address_emptiness_iterator(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterable[bool]:
        return self.address_emptiness_iterator_(
            data=self._data, empty_entry=self.default_entry, start=start, end=end
        )

    def append(
        self,
        new_data: torch.Tensor,
        allow_padding: bool = False,
        allow_reshaping: bool = False,
    ):
        self._data = self.get_appended(
            new_data=new_data,
            allow_padding=allow_padding,
            allow_reshaping=allow_reshaping,
        )

    def clear(self):
        self.clear_(block=self)

    def compress(self, start: int = 0, end: Optional[int] = None):
        if end is None:
            end = len(self.data)

        self._data[start:end] = self.get_compressed(start=start, end=end)

    def empty_indices_iterator(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterable[int]:
        return self.empty_indices_iterator_(
            data=self._data, empty_entry=self.default_entry, start=start, end=end
        )

    def from_tensor(self, tensor: torch.Tensor) -> "PyTorchBlock":
        return self.from_tensor_(tensor=tensor, default_value=self.default_value)

    def get_appended(
        self,
        new_data: torch.Tensor,
        allow_padding: bool = False,
        allow_reshaping: bool = False,
    ):
        return self.append_(
            data=self._data,
            new_data=new_data,
            allow_padding=allow_padding,
            allow_reshaping=allow_reshaping,
        )

    def get_compressed(self, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
        return self.compress_(
            data=self._data, empty_entry=self.default_entry, start=start, end=end
        )

    def get_wiped(self) -> torch.Tensor:
        return self.wipe_(data=self._data, default_value=self.default_value)

    def non_empty_indices_iterator(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterable[int]:
        return self.non_empty_indices_iterator_(
            data=self._data, empty_entry=self.default_entry, start=start, end=end
        )

    def wipe(self):
        self._data = self.get_wiped()

    # endregion

    # region Dunder methods
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fixed_str = WHITESPACE_RE.sub(" ", str(self).strip())

        return f"{self.__class__.__name__} <{self.dtype} on {self.device}> {fixed_str}"

    def __str__(self):
        return str(self.data.data.cpu().numpy())

    # endregion
