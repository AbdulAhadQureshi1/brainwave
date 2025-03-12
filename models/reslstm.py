import torch
import torch.nn as nn
from models.resnet import ResNetV1dCustom

class TemporalResNet(nn.Module):
    def __init__(self, num_classes=1, num_windows=1):
        super(TemporalResNet, self).__init__()
        self.resnet = ResNetV1dCustom(in_channels=19, depth=50)  # Specify 19 input channels
        self.hidden_size = 256
        self.num_layers = 2

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # BiLSTM layers
        self.bilstm1 = nn.LSTM(input_size=self.resnet.feat_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=self.hidden_size * 2, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        # Classification layers
        self.fc1 = nn.Linear(num_windows * 2 * self.hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_windows, num_time_samples, num_channels)
        batch_size, num_windows, num_time_samples, num_channels = x.size()

        # Reshape input to fit ResNet
        x = x.view(batch_size * num_windows, num_channels, num_time_samples, 1)  # Reshape to (batch_size * num_windows, num_channels, num_time_samples, 1)

        # Pass through ResNet
        features = self.resnet(x)[0]  # Output shape: (batch_size * num_windows, feat_dim, 1, 1)
        features = self.global_avgpool(features)
        features = features.view(batch_size * num_windows, 2048)  # Flatten

        # Reshape back to (batch_size, num_windows, feat_dim)
        features = features.view(batch_size, num_windows, 2048)

        # Pass through BiLSTM layers: in (4, 5, 2048)
        lstm_out1, _ = self.bilstm1(features) 
        lstm_out2, _ = self.bilstm2(lstm_out1)

        # Flatten the output for classification
        lstm_out2 = lstm_out2.contiguous().view(batch_size, -1)
        
        # Pass through classification layers
        out = self.fc1(lstm_out2)
        out = self.fc2(out)

        return torch.sigmoid(out).squeeze()
