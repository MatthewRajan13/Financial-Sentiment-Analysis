import torch.nn as nn


class RNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 256,
            n_layers: int = 2,
            n_classes: int = 3
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.float()
        outputs, _ = self.rnn(x)
        outputs = self.fc(outputs)
        return outputs
