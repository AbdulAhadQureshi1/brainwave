import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

# ----------------- CNN + Transformer for Temporal Features -----------------
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

class CNNTransformerModel(nn.Module):
    def __init__(self, num_nodes=19):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(num_nodes, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(512, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling → [batch, 32, 1]
        )
        self.projection = nn.Linear(32, 256)
        self.positional_encoding = PositionalEncoding(256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
    
    def forward(self, x):
        batch_size, num_windows, num_time_steps, num_nodes = x.size()
        x = x.view(-1, num_time_steps, num_nodes).permute(0, 2, 1)  # [batch*windows, 19, 1000]
        x = self.cnn_layers(x).squeeze(-1)  # [batch*windows, 32]
        x = x.view(batch_size, num_windows, 32)  # [batch, windows, 32]
        x = self.projection(x)  # [batch, windows, 256]
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return x  # [batch, windows, 256]

# ----------------- GAT for Spatial Features -----------------
class GATModule(nn.Module):
    def __init__(self, in_channels=1000, hidden_channels=64, out_channels=256, num_nodes=19, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, graph_batch):
        x, edge_index = graph_batch.x, graph_batch.edge_index
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index)
        return x  

# ----------------- Cross-Attention for Feature Fusion -----------------
class CrossAttention(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
    
    def forward(self, spatial_features, temporal_features):
        """
        spatial_features: [batch, windows, num_nodes, 256]
        temporal_features: [batch, windows, 256]
        """
        # Project spatial features to match Transformer shape
        batch, windows, num_nodes, feat_dim = spatial_features.shape
        spatial_features = spatial_features.view(batch, windows, num_nodes, feat_dim)  # [batch, windows, num_nodes, 256]

        # Cross-attention between spatial (queries) and temporal (keys/values)
        attended_spatial, _ = self.multihead_attn(spatial_features, temporal_features, temporal_features)
        return attended_spatial  # [batch, windows, num_nodes, 256]

# ----------------- Final Hybrid Model -----------------
class CTGATwithCrossAttention(nn.Module):
    def __init__(self, num_nodes=19, num_classes=1):
        super().__init__()
        self.cnn_transformer = CNNTransformerModel(num_nodes)
        self.gat = GATModule()
        self.cross_attention = CrossAttention(dim=256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, eeg_data, edge_index, batch):
        """
        eeg_data: [batch_size, num_windows, num_time_steps=1000, num_nodes=19]
        edge_index: Edge connections for GAT
        batch: Batch assignment for GAT
        """
        batch_size, num_windows, num_time_steps, num_nodes = eeg_data.shape

        # 1️⃣ Extract temporal features from CNN-Transformer
        temporal_features = self.cnn_transformer(eeg_data)  # [batch, windows, 256]

        
        # 2️⃣ Create graph batch
        graphs = []
        for b in range(batch_size):
            for w in range(num_windows):
                node_features = eeg_data[b, w].T  
                graphs.append(Data(x=node_features, edge_index=edge_index.clone()))  

        graph_batch = Batch.from_data_list(graphs)

        # 3️⃣ Extract spatial features using GAT
        spatial_features = self.gat(graph_batch)
        spatial_features = spatial_features.view(batch_size, num_windows, num_nodes, -1)  # [batch, windows, num_nodes, 256]

        # 4️⃣ Cross-Attention Fusion
        fused_features = self.cross_attention(spatial_features, temporal_features)  # [batch, windows, num_nodes, 256]

        # 5️⃣ Pooling & Classification
        fused_features = fused_features.mean(dim=2)  # Mean over nodes → [batch, windows, 256]
        fused_features = fused_features.mean(dim=1)  # Mean over windows → [batch, 256]

        return self.fc(fused_features)  # Output shape: [batch, num_classes]