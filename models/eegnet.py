import torch
import torch.nn as nn

# Updated parameters for your data
fs = 100                  # Your sampling frequency
channel = 19              # Number of electrodes (19)
num_input = 1             # Input channels (treat each window separately)
num_class = 1             # Number of classes
signal_length = 2000      # Time samples per window

# Architecture parameters
F1 = 8                    # Temporal filters
D = 3                     # Depth multiplier
F2 = D * F1               # Pointwise filters (24)

# Kernel sizes (recalculated for fs=100)
kernel_size_1 = (1, round(fs / 2))       # (1, 50)
kernel_size_2 = (channel, 1)             # Spatial filter across 19 electrodes
kernel_size_3 = (1, round(fs / 8))       # (1, 13)
kernel_size_4 = (1, 1)

# Padding calculations
def get_padding(kernel_size):
    return (kernel_size[0] // 2, (kernel_size[1] - 1) // 2)

kernel_padding_1 = (0, (kernel_size_1[1] - 1) // 2 - 1)  # (0, 24)
kernel_padding_3 = get_padding(kernel_size_3)             # (0, 6)

# Pooling and dropout
kernel_avgpool_1 = (1, 4)
kernel_avgpool_2 = (1, 4)
dropout_rate = 0.2

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer 1: Temporal convolution
        self.conv2d = nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Layer 2: Spatial convolution
        self.depthwise_conv = nn.Conv2d(F1, F1*D, kernel_size_2, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_avgpool_1)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Layer 3: Separable convolution
        self.sep_conv_depth = nn.Conv2d(F2, F2, kernel_size_3, 
                                       padding=kernel_padding_3, groups=F2)
        self.sep_conv_point = nn.Conv2d(F2, F2, kernel_size_4)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d(kernel_avgpool_2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Layer 4: Classification
        self.flatten = nn.Flatten()
        
        # Calculate the correct input size for the dense layer
        with torch.no_grad():
            x = torch.randn(1, num_input, channel, signal_length)
            x = self.bn1(self.conv2d(x))
            x = self.bn2(self.depthwise_conv(x))
            x = self.avgpool1(x)
            x = self.sep_conv_depth(x)
            x = self.sep_conv_point(x)
            x = self.avgpool2(x)
            self.fc_in_features = x.view(-1).shape[0]
            
        self.dense = nn.Linear(self.fc_in_features, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch, num_windows, time, channels)
        # Reshape to: (batch*num_windows, 1, channels, time)
        original_shape = x.shape
        x = x.view(-1, original_shape[2], original_shape[3]).unsqueeze(1)  # New shape: (B*W, 1, C, T)
        x = x.permute(0, 1, 3, 2)  # (B*W, 1, T, C) -> (B*W, 1, C, T) if needed
        
        # Layer 1
        x = self.bn1(self.conv2d(x))
        
        # Layer 2
        x = self.bn2(self.depthwise_conv(x))
        x = self.elu(x)
        x = self.dropout1(self.avgpool1(x))
        
        # Layer 3
        x = self.sep_conv_depth(x)
        x = self.sep_conv_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(self.avgpool2(x))
        
        # Layer 4
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        
        # Reshape back to (batch, num_windows, num_class) and average
        x = x.view(original_shape[0], original_shape[1], -1)
        x = x.mean(dim=1)  # Average over windows
        
        return x.squeeze()