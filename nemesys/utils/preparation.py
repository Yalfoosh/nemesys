import math
from typing import Tuple

import torch

from nemesys.utils.conversion import IterableConversion
from nemesys.utils.exceptions import ModeException


class PyTorchTensorPreparation:
    _pad_modes = {
        "begin",
        "center",
        "end",
    }

    @staticmethod
    def as_padded(
        tensor: torch.Tensor,
        shape: Tuple[int],
        n_to_pad: int,
        pad_mode: str = "end",
    ) -> torch.Tensor:
        n_to_pad = max(0, n_to_pad)

        if n_to_pad != 0:
            if pad_mode == "begin":
                left_pad = n_to_pad
            elif pad_mode == "center":
                left_pad = n_to_pad // 2
            elif pad_mode == "end":
                left_pad = 0
            else:
                pad_strings = IterableConversion.to_readable_string(
                    iterable=PyTorchTensorPreparation._pad_modes, last_prefix=" or "
                )

                raise ModeException(
                    f"Invalid pad mode: {pad_mode}; must be one of {pad_strings}"
                )

            right_pad = n_to_pad - left_pad

            left_pad = torch.zeros(
                left_pad,
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=False,
            )
            right_pad = torch.zeros(
                right_pad,
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=False,
            )

            tensor = torch.cat((left_pad, tensor.flatten(), right_pad)).reshape(
                (-1, *shape)
            )

        return tensor

    @staticmethod
    def for_block(tensor: torch.Tensor, dtype: torch.dtype, device: torch.device):
        tensor.requires_grad = False

        return tensor.to(device=device, dtype=dtype, non_blocking=False, copy=False)

    @staticmethod
    def for_block_insertion(
        tensor: torch.Tensor,
        base_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        tensor = PyTorchTensorPreparation.for_block(
            tensor=tensor, dtype=dtype, device=device
        )

        if tensor.shape == base_shape:
            tensor = tensor.unsqueeze(dim=0)

        if tensor.shape[1:] != base_shape:
            shape_size = math.prod(base_shape)

            n_to_pad = shape_size - (tensor.numel() % shape_size)
            n_to_pad %= shape_size

            if n_to_pad != 0:
                tensor = PyTorchTensorPreparation.as_padded(
                    tensor=tensor,
                    shape=base_shape,
                    n_to_pad=n_to_pad,
                    pad_mode="end",
                )

            if tensor.shape[1:] != base_shape:
                tensor = tensor.reshape((-1, base_shape))

        return tensor

