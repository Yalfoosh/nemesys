import torch


class PyTorchSynthesiserLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, content_key="content"):
        super().__init__()

        self._linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self._content_key = content_key

    @property
    def content_key(self):
        return self._content_key

    def forward(self, inputs):
        # (n, batch_size, ..., in_features)
        inputs = torch.stack(
            tuple(x[self._content_key] for x in inputs.values()), dim=0
        )
        # (batch_size, ..., in_features)
        inputs = torch.sum(inputs, dim=0)

        return {"content": self._linear(inputs)}
