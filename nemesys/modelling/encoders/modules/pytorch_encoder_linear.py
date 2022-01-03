import torch
import torch.nn
import torch.nn.functional


class PyTorchEncoderLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        content_key="content",
    ):
        super().__init__()

        self._linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self._content_key = content_key

    @property
    def content_key(self):
        return self._content_key

    def forward(self, inputs):
        inputs = inputs[self._content_key]

        return {"content": self._linear(inputs)}
