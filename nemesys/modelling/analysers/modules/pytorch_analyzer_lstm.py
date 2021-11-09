from typing import Dict, Iterable, List

import torch
import torch.nn
import torch.nn.functional


class PyTorchAnalyzerLSTM(torch.nn.Module):
    def __init__(
        self,
        class_names: Iterable[str],
        *args,
        **kwargs,
    ):
        super().__init__()

        # Ensure class names are unique, but keep order
        self._class_names = list(dict.fromkeys(str(x) for x in class_names))
        self._lstm = torch.nn.LSTM(*args, **kwargs)

        # If number of classes is 1, we don't need a Linear layer
        if len(self._class_names) == 1:
            self._classifier = lambda x: torch.full(
                size=(*x.shape[:2], len(self._class_names)),
                fill_value=1.0,
                layout=x.layout,
                dtype=x.dtype,
                device=x.device,
                requires_grad=False,
            )
        else:
            self._classifier = torch.nn.Linear(
                in_features=self._lstm.hidden_size,
                out_features=len(self._class_names),
                bias=True,
            )

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def forward(self, inputs: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        batch_size = inputs.shape[0 if self._lstm.batch_first else 1]

        # (batch_size/seq_len, seq_len/batch_size, hidden_size)
        lstm_out, _ = self._lstm(
            inputs,
            (
                torch.zeros(1, batch_size, self._lstm.hidden_size),
                torch.zeros(1, batch_size, self._lstm.hidden_size),
            ),
        )

        # (batch_size/seq_len, seq_len/batch_size, len(self._class_names))
        class_logits = self._classifier(lstm_out)
        class_probabilities = torch.nn.functional.softmax(class_logits, dim=-1)

        # (batch_size/seq_len, seq_len/batch_size)
        classifications = torch.argmax(class_probabilities, dim=-1)

        # Dict[class_name, Dict[tensor_name, tensor]]
        return {
            class_name: {"output": lstm_out[classifications == i]}
            for i, class_name in enumerate(self._class_names)
        }
