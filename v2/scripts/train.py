import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from data.dataset import EEGDataset
from models.model import CombinedModel

# Hyperparameters
batch_size = 20
lr = 1e-4
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
experiment_name = 'hybrid_model_exp_1'

os.makedirs(f'./experiments/{experiment_name}', exist_ok=True)
model_path = f'./experiments/{experiment_name}/best_model.pth'

train_dataset = EEGDataset(data_dir="../preprocessed_data", labels_json_path="../metadata.json")

train_ratio = 0.8
val_ratio = 1 - train_ratio

train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=4, shuffle=False)

# Model setup
resnet_config = {
    'in_channels': 19,
    'base_filters': 64,
    'kernel_size': 17,
    'stride': 2,
    'groups': 1,
    'n_block': 4,
    'downsample_gap': 2,
    'increasefilter_gap': 4,
    'use_bn': True,
    'use_do': True,
    # 'input_length': 1000,
    'verbose': False
}
transformer_config = {
    'output_dim': 1,
    'hidden_dim': 64,
    'num_heads': 4,
    'key_query_dim': 32,
    'intermediate_dim': 128
}
model = CombinedModel(resnet_config, transformer_config).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
best_val_loss = float('inf')

# Metrics
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels.float()).item()
    
    # Calculate metrics
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)

    # Save checkpoint every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'./experiments/{experiment_name}/epoch_{epoch+1}.pth')

# Save final metrics and plots
def save_plots(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./experiments/{experiment_name}/loss_curve.png')
    plt.close()

save_plots(train_losses, val_losses)
print(f"Training completed. Best model saved to {model_path}")