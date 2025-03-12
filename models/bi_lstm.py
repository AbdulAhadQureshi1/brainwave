import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, input_channels=19, lstm_hidden_size=128, lstm_num_layers=2):
        super(BiLSTMModel, self).__init__()

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_channels,  # input_size is the number of channels
            hidden_size=lstm_hidden_size,  # The size of hidden state
            num_layers=lstm_num_layers,  # Number of LSTM layers
            batch_first=True,  # Input format is (batch_size, seq_len, input_size)
            bidirectional=True  # Bidirectional LSTM
        )
        
        # Fully connected layer for classification
        self.fc1 = nn.Linear(2 * lstm_hidden_size, 1)  # 2 for bidirectional

    def forward(self, x):
        """
        x: Tensor of shape (num_windows, time_steps, input_channels)
        """
        print(x.shape)
        batch_size, time_steps, input_channels = x.shape

        # Pass through the BiLSTM layer
        lstm_out, _ = self.lstm(x)  # Shape: (num_windows, time_steps, 2 * lstm_hidden_size)

        # Take the output of the last time step for each window
        lstm_out = lstm_out[:, -1, :]  # Shape: (num_windows, 2 * lstm_hidden_size)

        # Pass through the fully connected layer for classification
        output = torch.sigmoid(self.fc1(lstm_out)).squeeze()  # Shape: (num_windows)

        return output