import math
from typing import Tuple

import torch

from nemesys.utils.exceptions import PaddingException, ReshapingException


class TensorPreparation:
    @staticmethod
    def as_padded(
        tensor: torch.Tensor, shape: Tuple[int], n_to_pad: int
    ) -> torch.Tensor:
        n_to_pad = max(0, n_to_pad)

        if n_to_pad != 0:
            tensor = torch.cat(
                (
                    tensor.flatten(),
                    torch.zeros(
                        n_to_pad,
                        dtype=tensor.dtype,
                        device=tensor.device,
                        requires_grad=False,
                    ),
                )
            ).reshape((-1, *shape))

        return tensor

    @staticmethod
    def for_block(tensor: torch.Tensor, dtype: torch.dtype, device: torch.device):
        tensor.requires_grad = False

        return tensor.to(device=device, dtype=dtype, non_blocking=False, copy=False)

    @staticmethod
    def for_block_insertion(
        tensor: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        base_shape: Tuple[int],
        allow_padding: bool = False,
        allow_reshaping: bool = False,
    ):
        tensor = TensorPreparation.for_block(tensor=tensor, dtype=dtype, device=device)

        if tensor.shape == base_shape:
            tensor = tensor.unsqueeze(dim=0)

        if tensor.shape[1:] != base_shape:
            shape_size = math.prod(base_shape)

            n_to_pad = shape_size - (tensor.numel() % shape_size)
            n_to_pad %= shape_size

            if n_to_pad != 0:
                if allow_padding:
                    tensor = TensorPreparation.as_padded(
                        tensor=tensor, shape=base_shape, n_to_pad=n_to_pad
                    )
                else:
                    raise PaddingException(
                        "Tensor needs padding, but `allow_padding` is set to False"
                    )

            if tensor.shape[1:] != base_shape:
                if allow_reshaping:
                    tensor = tensor.reshape((-1, base_shape))
                else:
                    raise ReshapingException(
                        "Tensor needs reshaping, but `allow_reshaping` is set to False"
                    )

        return tensor
