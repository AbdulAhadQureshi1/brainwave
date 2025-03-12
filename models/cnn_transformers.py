import torch
import torch.nn as nn

class CNNTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN expects input [N, EEG_channels=19, time_steps=1000]
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(19, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),  # Added
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(512, 128, 3, padding=1),
            nn.BatchNorm1d(128),  # Added
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 32, 3, padding=1),
            nn.BatchNorm1d(32),  # Added
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling to [N, 32, 1]
        )
        self.projection = nn.Linear(32, 256)
        self.positional_encoding = PositionalEncoding(256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.3)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)  # Deeper
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        batch_size, num_windows, num_time_steps, num_channels = x.size()
        x = x.view(-1, num_channels, num_time_steps)  # [N*10, 19, 1000]
        x = self.cnn_layers(x).squeeze(-1)  # [N*10, 32]
        x = x.view(batch_size, num_windows, 32)
        x = self.projection(x)  # [batch, num_windows, 256]
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return (self.classifier(x)).squeeze(-1)  # Remove sigmoid if using BCEWithLogitsLoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # Match sequence length