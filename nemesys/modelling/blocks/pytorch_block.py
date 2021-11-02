from typing import Any, Iterable, Tuple, Union, Optional

import torch

from nemesys.modelling.blocks.block import ShapedBlock
from nemesys.utils.conversion import (
    DeviceConversion,
    DtypeConversion,
    IterableConversion,
)
from nemesys.utils.exceptions import ModeException
from nemesys.utils.preparation import PyTorchTensorPreparation
from nemesys.utils.re import WHITESPACE_RE


class PyTorchBlock(ShapedBlock):
    _init_methods = {
        "tensor",
    }

    def __init__(
        self,
        base_shape: Union[int, Iterable[int]],
        dtype: Union[str, torch.dtype] = "float32",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(base_shape=base_shape)

        self._dtype = DtypeConversion.to_torch(dtype)
        self._device = DeviceConversion.to_torch(device)

        self._data = self.get_defaulted()

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

    # endregion

    # region Cast methods
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "PyTorchBlock":
        base_shape = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[1:]

        block = PyTorchBlock(
            base_shape=base_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        block.write(content=tensor)

        return block

    # endregion

    # region Static methods
    @staticmethod
    def prepare_default(
        base_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(
            size=(0, *base_shape), dtype=dtype, device=device, requires_grad=False
        )

    @staticmethod
    def prepare_read(content: torch.Tensor, key: Optional[int]) -> torch.Tensor:
        if key is None:
            return content
        else:
            return content[key]

    @staticmethod
    def prepare_write(
        content: Optional[torch.Tensor],
        base_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if content is None:
            new_content = PyTorchBlock.prepare_default(
                base_shape=base_shape,
                dtype=dtype,
                device=device,
            )
        else:
            new_content = PyTorchTensorPreparation.for_block_insertion(
                tensor=content,
                base_shape=base_shape,
                dtype=dtype,
                device=device,
            )

        return new_content

    # endregion

    # region Generation methods
    def get_defaulted(self) -> torch.Tensor:
        return self.prepare_default(
            base_shape=self._base_shape, dtype=self._dtype, device=self._device
        )

    def get_read(self, key: Optional[Iterable[int]]) -> torch.Tensor:
        return self.prepare_read(content=self._data, key=key)

    def get_written(self, content: Optional[torch.Tensor]) -> torch.Tensor:
        return self.prepare_write(old_content=self._data, new_content=content)

    # endregion

    # region Block implementation
    @staticmethod
    def init_from(content: Any, cast_method: Optional[str] = None) -> "PyTorchBlock":
        if cast_method is None:
            if isinstance(content, torch.Tensor):
                cast_method = "tensor"
            else:
                cast_method = None

        if cast_method not in PyTorchBlock._init_methods:
            cast_method_strings = IterableConversion.to_readable_string(
                iterable=PyTorchBlock._init_methods, last_prefix=" or "
            )

            raise ModeException(
                f"Invalid cast method: {cast_method}; must be one of "
                f"{cast_method_strings}"
            )

        if cast_method == "tensor":
            return PyTorchBlock.from_tensor(tensor=content)

    def default(self) -> "PyTorchBlock":
        return PyTorchBlock.from_tensor(tensor=self.get_defaulted())

    def read(self, key: Optional[Iterable[int]] = None) -> torch.Tensor:
        return self.get_read(key=key)

    def write(self, content: Optional[torch.Tensor] = None):
        return self.get_written(content=content)

    # endregion

    # region Other methods
    # TODO
    """
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

    def address_emptiness_iterator(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterable[bool]:
        return self.address_emptiness_iterator_(
            data=self._data, empty_entry=self.default_entry, start=start, end=end
        )

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

    def get_compressed(self, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
        return self.compress_(
            data=self._data, empty_entry=self.default_entry, start=start, end=end
        )

    def non_empty_indices_iterator(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterable[int]:
        return self.non_empty_indices_iterator_(
            data=self._data, empty_entry=self.default_entry, start=start, end=end
        )
    """

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
