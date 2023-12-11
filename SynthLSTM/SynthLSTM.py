# SynthLSTM.py

import torch
import torch.nn as nn

class SynthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SynthLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Transform lstm_out to have same shape as target data
        out = self.fc(lstm_out)  # Apply linear layer to each time step
        return out
