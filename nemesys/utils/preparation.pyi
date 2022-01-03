from typing import Set, Tuple

import torch

class PyTorchTensorPreparation:
    _pad_modes: Set[str]
    @staticmethod
    def as_padded(
        tensor: torch.Tensor,
        shape: Tuple[int],
        n_to_pad: int,
        pad_mode: str,
    ) -> torch.Tensor: ...
    @staticmethod
    def for_block(
        tensor: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor: ...
    @staticmethod
    def for_block_insertion(
        tensor: torch.Tensor,
        base_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor: ...
