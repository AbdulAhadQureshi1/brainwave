import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class EEGTransformer(nn.Module):
    def __init__(self, num_channels, num_timepoints, output_dim,
                 hidden_dim, num_heads, key_query_dim,
                 hidden_ffn_dim, intermediate_dim, ffn_output_dim):
        super(EEGTransformer, self).__init__()

        # Positional Encoding
        self.positional_encoding = torch.zeros(num_channels, num_timepoints)
        for j in range(num_channels):
            for k in range(num_timepoints):
                if j % 2 == 0:
                    self.positional_encoding[j][k] =\
                        torch.sin(torch.tensor(k) / (10000 ** (torch.tensor(j) / num_channels)))
                else:
                    self.positional_encoding[j][k] =\
                        torch.cos(torch.tensor(k) / (10000 ** ((torch.tensor(j) - 1) / num_channels)))

        # Multi-Head Self Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=num_channels,
                                                    num_heads=num_heads)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(num_channels, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, ffn_output_dim)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)

        # Classifier
        self.classifier = nn.Linear(num_channels * num_timepoints, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # Input Standardization
        mean = X.mean(dim=2, keepdim=True)
        std  = X.std(dim=2, keepdim=True)
        X_hat = (X - mean) / (std + 1e-5)  # epsilon to avoid division by zero

        # Add Positional Encoding
        X_tilde = X_hat + self.positional_encoding.to(X.device)

        # Reshape for multi-head self attention: (seq_len, batch_size, embed_dim)
        X_tilde = X_tilde.permute(2, 0, 1)

        # Multi-Head Self Attention
        attn_output, _ = self.multihead_attn(X_tilde, X_tilde, X_tilde)

        # Reshape back and Apply Layer Norm
        # attn_output = attn_output.permute(1, 2, 0)  # Reshape: (batch_size, embed_dim, seq_len)
        X_ring = torch.stack([self.norm1(a) for a in attn_output], dim=1)

        # Position-wise Feed-Forward Networks
        ff_output = self.ffn(X_ring)
        O = self.norm2(ff_output + X_ring)

        # Classifier
        # Flatten and classify
        O_flat = O.view(O.size(0), -1)  # Flatten the tensor
        output = self.classifier(O_flat)

        return self.sigmoid(output).squeeze()
