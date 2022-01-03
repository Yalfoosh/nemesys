import torch

from nemesys.modelling.blocks.shaped_block import ShapedBlock
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

    def __init__(self, base_shape, dtype="float32", device="cpu"):
        super().__init__(base_shape=base_shape)

        self._dtype = DtypeConversion.to_torch(dtype)
        self._device = DeviceConversion.to_torch(device)

        self._data = self.get_defaulted()

    # region Properties
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    # endregion

    # region Cast methods
    @staticmethod
    def from_tensor(tensor):
        base_shape = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[1:]

        block = PyTorchBlock(
            base_shape=base_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        block.data = tensor

        return block

    # endregion

    # region Static methods
    @staticmethod
    def prepare_data(content):
        return content.clone()

    @staticmethod
    def prepare_default(base_shape, dtype, device):
        return torch.empty(
            size=(0, *base_shape), dtype=dtype, device=device, requires_grad=False
        )

    @staticmethod
    def prepare_tensor(content, base_shape, dtype, device):
        if content is None:
            content = PyTorchBlock.prepare_default(
                base_shape=base_shape,
                dtype=dtype,
                device=device,
            )
        else:
            content = PyTorchTensorPreparation.for_block_insertion(
                tensor=content,
                base_shape=base_shape,
                dtype=dtype,
                device=device,
            )

        return content

    # endregion

    # region Generation methods
    def get_data(self):
        return self.prepare_data(content=self._data)

    def get_defaulted(self):
        return self.prepare_default(
            base_shape=self._base_shape, dtype=self._dtype, device=self._device
        )

    def get_tensor(self, content):
        return self.prepare_tensor(
            content=content,
            base_shape=self._base_shape,
            dtype=self._dtype,
            device=self._device,
        )

    # endregion

    # region Block implementation
    @property
    def data(self):
        return self.get_data()

    @data.setter
    def data(self, value):
        self._data = self.get_tensor(content=value)

    @staticmethod
    def init_from(content, cast_method=None):
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

    def clone(self):
        return self.init_from(content=self._data)

    def default(self):
        return PyTorchBlock.from_tensor(tensor=self.get_defaulted())

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
