import math
from typing import Iterable, Union

import torch

from nemesys.utils.spans import get_spans_from_indices


class Block:
    def __init__(
        self,
        base_shape: Union[int, Iterable[int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        if isinstance(base_shape, int):
            base_shape = (base_shape,)

        self._base_shape = tuple(base_shape)

        if any((not isinstance(size, int) or size < 1) for size in self.base_shape):
            raise ValueError("Base shape must be an iterable of positive integers")

        self._n_base_values = math.prod(self.base_shape)

        if not isinstance(dtype, torch.dtype):
            raise ValueError("Dtype must be a torch.dtype")

        self._dtype = dtype

        if not isinstance(device, torch.device):
            try:
                device = str(device)
            except TypeError:
                raise TypeError("Device must be castable to str")

            try:
                device = torch.device(device)
            except Exception:
                raise ValueError("Device must be a valid torch.device")

        self._device = device
        self._content = self.reserve(size=0)

    # region Properties
    @property
    def base_shape(self) -> Iterable[int]:
        return self._base_shape

    @property
    def content(self) -> torch.Tensor:
        return self._content

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def n_base_values(self) -> int:
        return self._n_base_values

    @property
    def shape(self) -> Iterable[int]:
        return self._content.shape

    # endregion

    def allocate(self, size: int = 1):
        size = max(0, size)

        if size != 0:
            self._content = torch.cat((self._content, self.reserve(size=size)))

    def append(
        self,
        data: torch.Tensor,
        allow_reshape: bool = False,
        allow_pad: bool = False,
    ):
        data = self.prepare_tensor_for_insert(
            data=data, allow_reshape=allow_reshape, allow_pad=allow_pad
        )

        self._content = torch.cat((self._content, data))

    def clear(self):
        self.deallocate(size=len(self._content))

    def compress(self):
        non_empty_addresses = self.get_non_empty_addresses()
        non_empty_spans = get_spans_from_indices(
            indices=non_empty_addresses, is_sorted=True
        )

        self._content = torch.cat(
            [self._content[begin:end] for begin, end in non_empty_spans]
        )

    def deallocate(self, size: int = 1):
        size = max(0, min(len(self._content), size))

        if size != 0:
            self._content = self._content[:-size]

    def get_empty_addresses(self):
        empty_row = self.reserve(size=1)[0]

        return [i for i, row in enumerate(self._content) if torch.equal(row, empty_row)]

    def get_non_empty_addresses(self):
        empty_row = self.reserve(size=1)[0]

        return [
            i for i, row in enumerate(self._content) if not torch.equal(row, empty_row)
        ]

    def pad_tensor(self, data: torch.Tensor, n_to_pad: int = 0) -> torch.Tensor:
        n_to_pad = max(0, n_to_pad)

        if n_to_pad != 0:
            data = torch.cat(
                (
                    data.flatten(),
                    torch.zeros(
                        n_to_pad,
                        dtype=data.dtype,
                        device=data.device,
                        requires_grad=False,
                    ),
                )
            ).reshape((-1, *self.base_shape))

        return data

    def prepare_tensor(self, data: torch.Tensor):
        if data.requires_grad:
            data.requires_grad = False

        return data.to(device=self.device, dtype=self.dtype)

    def prepare_tensor_for_insert(
        self, data: torch.Tensor, allow_reshape: bool = False, allow_pad: bool = False
    ) -> torch.Tensor:
        data = self.prepare_tensor(data=data)

        if data.shape == self.base_shape:
            data = data.unsqueeze(dim=0)

        if data.shape[1:] != self.base_shape:
            n_to_pad = self.n_base_values - (data.numel() % self.n_base_values)
            n_to_pad %= self.n_base_values

            if n_to_pad != 0:
                if allow_pad:
                    self.pad_tensor(data=data, n_to_pad=n_to_pad)
                else:
                    raise ValueError(
                        f"Data would need to have {n_to_pad} values padded, but "
                        "`allow_pad` is set to False"
                    )

            if data.shape[1:] != self.base_shape:
                if allow_reshape:
                    data = self.reshape_tensor(data=data)
                else:
                    raise ValueError(
                        "Data needs to be reshaped, but `allow_reshape` is set to False"
                    )

        return data

    def reserve(self, size: int = 1) -> torch.Tensor:
        return torch.zeros(
            size=(size, *self.base_shape),
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )

    def reshape_tensor(self, data: torch.Tensor) -> torch.Tensor:
        return data.reshape((-1, *self.base_shape))

    def wipe(self):
        self._content = torch.zeros(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )

    @staticmethod
    def from_tensor(
        data: torch.Tensor, allow_reshape: bool = False, allow_pad: bool = False
    ) -> torch.Tensor:
        base_shape = data.shape[0] if len(data.shape) == 1 else data.shape[1:]

        block = Block(base_shape=base_shape, dtype=data.dtype, device=data.device)
        block.append(data, allow_reshape=allow_reshape, allow_pad=allow_pad)

        return block

    # region Dunder
    def __getitem__(self, index: int):
        return self._content[index]

    def __len__(self):
        return len(self._content)

    # endregion
