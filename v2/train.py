import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from data.dataset import EEGDataset
from models.model import CombinedModel
from sklearn.metrics import roc_auc_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

def focal_loss_bce(logits: torch.Tensor,
                   targets: torch.Tensor,
                   alpha: float = 0.25,
                   gamma: float = 2.0,
                   reduction: str = 'mean') -> torch.Tensor:
    # Standard binary cross-entropy loss per element (no reduction)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # Convert logits -> probabilities
    p = torch.sigmoid(logits)

    # p_t is p if target=1 else (1-p)
    p_t = p * targets + (1 - p) * (1 - targets)

    # alpha_t is alpha if target=1 else (1-alpha)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    # Focal modulation factor: alpha_t * (1 - p_t)^gamma
    focal_factor = alpha_t * (1 - p_t).pow(gamma)

    # Combine focal factor with the bce loss
    loss = focal_factor * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def load_model(model_path, resnet_config, transformer_config, device):
    # Initialize the model
    model = CombinedModel(resnet_config, transformer_config).to(device)
    
    # Load the saved model state
    model.load_state_dict(torch.load(model_path))
    return model


# Hyperparameters
batch_size = 16
lr = 1e-4
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
experiment_name = 'hybrid_model_newdata_1'
old_model_path = './experiments/hybrid_model_newdata_1/best_model.pth'

os.makedirs(f'./experiments/{experiment_name}', exist_ok=True)
model_path = f'./experiments/{experiment_name}/best_model.pth'

train_dataset = EEGDataset(data_dir="./preprocessed_data", labels_json_path="./train_metadata.json")
val_dataset = EEGDataset(data_dir="./preprocessed_data", labels_json_path="./val_metadata.json")

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

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
    # 'input_length': 2000,
    'verbose': False
}
transformer_config = {
    'output_dim': 1,
    'hidden_dim': 32,
    'num_heads': 4,
    'key_query_dim': 32,
    'intermediate_dim': 128
}

# model = CombinedModel(resnet_config, transformer_config).to(device)
model = load_model(old_model_path,resnet_config, transformer_config, device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8])).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
best_val_loss = float('inf')

# Metrics
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_probs = []
    train_targets = []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        train_probs.extend(probs)
        train_targets.extend(targets)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels.float()).item()

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()

            val_probs.extend(probs)
            val_targets.extend(targets)

    # Metrics calculation
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    # Convert to numpy arrays
    train_probs = np.array(train_probs)
    train_preds = (train_probs > 0.5).astype(int)
    train_targets = np.array(train_targets)

    val_probs = np.array(val_probs)
    val_preds = (val_probs > 0.5).astype(int)
    val_targets = np.array(val_targets)

    # Train Metrics
    train_tp = ((train_preds == 1) & (train_targets == 1)).sum()
    train_fp = ((train_preds == 1) & (train_targets == 0)).sum()
    train_fn = ((train_preds == 0) & (train_targets == 1)).sum()
    train_tn = ((train_preds == 0) & (train_targets == 0)).sum()

    train_tpr = train_tp / (train_tp + train_fn + 1e-8)
    train_fpr = train_fp / (train_fp + train_tn + 1e-8)
    train_auc = roc_auc_score(train_targets, train_probs)

    # Val Metrics
    val_tp = ((val_preds == 1) & (val_targets == 1)).sum()
    val_fp = ((val_preds == 1) & (val_targets == 0)).sum()
    val_fn = ((val_preds == 0) & (val_targets == 1)).sum()
    val_tn = ((val_preds == 0) & (val_targets == 0)).sum()

    val_tpr = val_tp / (val_tp + val_fn + 1e-8)
    val_fpr = val_fp / (val_fp + val_tn + 1e-8)
    val_auc = roc_auc_score(val_targets, val_probs)

    # Print results
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    print(f'Train - TPR: {train_tpr:.4f} FPR: {train_fpr:.4f} AUC: {train_auc:.4f}')
    print(f'Val - TPR: {val_tpr:.4f} FPR: {val_fpr:.4f} AUC: {val_auc:.4f}')

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)

    # Save checkpoint every 10 epochs
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f'./experiments/{experiment_name}/epoch_{epoch+1}.pth')
    
    # scheduler.step(val_loss)

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