import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x)


class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels):
        super(Decoder, self).__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(input_channels, hidden_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # or ReLU, depending on your data range
        )

    def forward(self, x):
        return self.deconv_layers(x)


class AudioEncoderDecoder(nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels):
        super(AudioEncoderDecoder, self).__init__()
        self.encoder = Encoder(input_channels, hidden_size)
        self.decoder = Decoder(hidden_size * 4, hidden_size, output_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

