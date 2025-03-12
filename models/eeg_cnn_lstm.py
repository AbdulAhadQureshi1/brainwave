import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()  # Fixed constructor
        self.conv = nn.Conv1d(input_channels, num_filters, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EEGModel(nn.Module):
    def __init__(  # Fixed constructor
        self,
        input_channels=19,
        cnn_filters=64,
        cnn_kernel_size=3,
        cnn_stride=1,
        cnn_padding=1,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        fc_hidden_size=64,
    ):
        super(EEGModel, self).__init__()
        
        # Define CNNs
        self.num_cnns = 70
        self.cnns = nn.ModuleList(
            [CNNBlock(input_channels, cnn_filters, cnn_kernel_size, cnn_stride, cnn_padding) for _ in range(self.num_cnns)]
        )

        # Define LSTMs
        self.lstm1 = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(4 * lstm_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, 1)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_windows, time_steps, input_channels)
        """
        num_windows, time_steps, input_channels = x.shape

        cnn_features = []

        # Process each window through its corresponding CNN
        for i in range(self.num_cnns):
            if i < num_windows:  # Ensure the CNN index is within bounds
                window = x.permute(0, 2, 1)  # Reshape to (num_windows, input_channels, time_steps)
                cnn_features.append(self.cnns[i](window).mean(dim=2))  # Average across time

        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, num_windows, cnn_filters)

        # Split features into two groups
        features_5min = cnn_features[:, :30]  # First 30 CNNs
        features_remaining = cnn_features[:, 30:]  # Remaining CNNs

        # BiLSTM 1: Process 5-minute features
        lstm1_out, _ = self.lstm1(features_5min)  # (batch_size, 30, 2 * lstm_hidden_size)
        lstm1_out = lstm1_out.mean(dim=1)  # Average across time

        # BiLSTM 2: Process remaining features
        lstm2_out, _ = self.lstm2(features_remaining)  # (batch_size, remaining_windows, 2 * lstm_hidden_size)
        lstm2_out = lstm2_out.mean(dim=1)  # Average across time

        # Combine LSTM outputs
        combined = torch.cat([lstm1_out, lstm2_out], dim=1)  # (batch_size, 4 * lstm_hidden_size)

        # Fully connected layers
        fc1_out = F.relu(self.fc1(combined))
        output = torch.sigmoid(self.fc2(fc1_out)).squeeze()

        return output
