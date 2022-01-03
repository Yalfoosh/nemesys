import torch
import torch.nn
import torch.nn.functional


class PyTorchDecoderConv2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        self._conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, inputs):
        # (n_blocks, )
        inputs = inputs.get_all()
        # (n_entries, *base_shape)
        inputs = torch.cat(tuple(x.data for x in inputs), dim=0)
        # (1, in_channels, n_blocks, prod(base_shape))
        inputs = torch.reshape(
            inputs, shape=(1, self._conv.in_channels, inputs.shape[0], -1)
        )

        return {"content": self._conv(inputs)}
