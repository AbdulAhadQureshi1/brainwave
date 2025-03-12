import torch
import torch.nn as nn


import torch
import torch.nn as nn

class CNNBiLSTMModel(nn.Module):
    def __init__(self):
        super(CNNBiLSTMModel, self).__init__()

        # Define the CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=19, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(19)  # Reduces to a single feature per channel
        )

        # Define the Bi-LSTM layer
        self.bilstm = nn.LSTM(
            input_size=19,  # Feature size from CNN
            hidden_size=128,  # LSTM hidden size
            num_layers=2,   # Number of LSTM layers
            bidirectional=True,  # Bi-LSTM
            batch_first=True
        )

        self.fc1 = nn.Linear(2 * 128, 64) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch_size, 10, 1000, 19]

        batch_size, num_windows, num_time_steps, num_channels = x.size()

        # Reshape to combine batch_size and num_windows
        x = x.view(batch_size * num_windows, num_channels, num_time_steps)

        # Pass through CNN layers
        x = self.cnn_layers(x)

        x = x.view(batch_size, num_windows, 64, 19)

        x = x.mean(dim=2) 

        # Pass through Bi-LSTM
        lstm_out, _ = self.bilstm(x) 

        # Take mean across the time dimension (num_windows)
        lstm_out = lstm_out.mean(dim=1)  # Output shape: [batch_size, 2 * hidden_size]

        # Pass through the fully connected layers
        fc_out = self.fc1(lstm_out)
        fc_out = torch.relu(fc_out)
        output = torch.sigmoid(self.fc2(fc_out).squeeze(-1))  # Apply sigmoid activation, Output shape: [batch_size]

        return output
    