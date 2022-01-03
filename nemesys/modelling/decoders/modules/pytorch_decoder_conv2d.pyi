from typing import Dict, Tuple, Union

import torch

from nemesys.modelling.stores.pytorch_list_store import PyTorchListStore

class PyTorchDecoderConv2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        groups: Union[int, Tuple[int, int]],
        bias: bool,
        padding_mode: str,
    ): ...
    def forward(self, inputs: PyTorchListStore) -> Dict[str, torch.Tensor]: ...
