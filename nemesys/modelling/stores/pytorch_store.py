from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn

from nemesys.modelling.blocks.pytorch_block import PyTorchBlock
from nemesys.modelling.stores.store import Store
from nemesys.utils.re import WHITESPACE_RE


class ListStore(Store, torch.nn.Module):
    def __init__(
        self,
        blocks: Iterable[PyTorchBlock] = tuple(),
        allow_padding: bool = False,
        allow_reshaping: bool = False,
    ):
        super().__init__()

        if isinstance(blocks, ListStore):
            self._blocks = list(blocks.blocks)
        else:
            self._blocks = list(blocks)

        self._allow_padding = allow_padding
        self._allow_reshaping = allow_reshaping

    # region Properties
    @property
    def blocks(self) -> Iterable[PyTorchBlock]:
        return self._blocks

    @property
    def allow_padding(self) -> bool:
        return self._allow_padding

    @property
    def allow_reshaping(self) -> bool:
        return self._allow_reshaping

    # endregion

    def forward(
        self,
        inputs: Iterable[Tuple[int, torch.Tensor]],
        allow_padding: Optional[bool] = None,
        allow_reshaping: Optional[bool] = None,
    ):
        for i, tensor in inputs:
            self._blocks[i].append(
                new_data=tensor,
                allow_padding=self.allow_padding
                if allow_padding is None
                else allow_padding,
                allow_reshaping=self.allow_reshaping
                if allow_reshaping is None
                else allow_reshaping,
            )

    # region Dunder methods
    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self):
        fixed_str = WHITESPACE_RE.sub(" ", str(self).strip())

        return f"{self.__class__.__name__}({len(self)} blocks) {fixed_str}"

    def __str__(self):
        return str(np.array([block.data.cpu().numpy() for block in self.blocks]))

    # endregion
