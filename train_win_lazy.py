import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models import CombinedModel
from data.loaders.eeg_dataset_win_lazy import create_dataloader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from utils import get_patients
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os
import logging
import colorlog
from typing import Tuple, Dict

# Setup colored logging
def setup_logger():
    """Configure colored logging"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger = colorlog.getLogger('eeg_training')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

class FPRPenaltyLoss(nn.Module):
    """Custom loss function with FPR penalty"""
    def __init__(self, base_criterion=nn.BCELoss(), fpr_target=0.05, fpr_weight=2.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.fpr_target = fpr_target
        self.fpr_weight = fpr_weight
    
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_criterion(outputs, labels)
        
        # Calculate FPR penalty
        with torch.no_grad():
            predictions = (outputs > 0.5).float()
            fp = torch.sum((predictions == 1) & (labels == 0))
            tn = torch.sum((predictions == 0) & (labels == 0))
            current_fpr = fp / (fp + tn + 1e-7)
        
        # Add FPR penalty if above target
        fpr_penalty = torch.relu(current_fpr - self.fpr_target) * self.fpr_weight
        total_loss = base_loss + fpr_penalty
        
        return total_loss

def calculate_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate all relevant metrics including TPR and FPR"""
    fpr, tpr, _ = roc_curve(labels, preds)
    
    # Find TPR at FPR ≤ 0.05
    valid_idx = np.where(fpr <= 0.05)[0]
    tpr_at_fpr5 = tpr[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
    
    return {
        'acc': accuracy_score(labels, (preds > 0.5)),
        'auc': roc_auc_score(labels, preds),
        'fpr': fpr.mean(),
        'tpr': tpr.mean(),
        'tpr_at_fpr5': tpr_at_fpr5
    }

# Hyperparameters
batch_size = 20
lr = 1e-4
epochs = 50
records_per_patient = -1 # -1 for everything, otherwise specify numbers
max_num_patients = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
experiment_name = 'trans_resnet_16b_460'

logger.info(f'Using device: {device}')

# Create directory if it doesn't exist
os.makedirs(f'./experiments/results/{experiment_name}', exist_ok=True)
os.makedirs(f'./experiments/logs/{experiment_name}', exist_ok=True)
os.makedirs(f'./experiments/models/{experiment_name}', exist_ok=True)

# Set Paths
results_path = f'./experiments/results/{experiment_name}'
logs_path = f'./experiments/logs/{experiment_name}'
models_path = f'./experiments/models/{experiment_name}'

# Prepare dataset
training_dir = '/media/brainwave/2Tb HDD/physionet.org/files/i-care/2.1/training'
training_data, validation_data = get_patients(
    training_dir,
    prototype=True,
    per_class=180,
    val_split=0.2,
    max_num_patients=max_num_patients
)

if len(training_data) == 0:
    logger.error("No patient data found")
    exit()

train_loader, val_loader = create_dataloader(
    training_data,
    training_dir,
    records_per_patient=12,
    batch_size=batch_size,
    predict='outcome'
)

resnet_config = {
    'in_channels': 19,
    'base_filters': 64,
    'kernel_size': 17,
    'stride': 2,
    'groups': 1,
    'n_block': 16,
    'downsample_gap': 2,
    'increasefilter_gap': 4,
    'use_bn': True,
    'use_do': True,
    'verbose': False
}

transformer_config = {
    'output_dim': 1,  # Binary classification
    'hidden_dim': 64,
    'num_heads': 4,
    'key_query_dim': 32,
    'intermediate_dim': 128
}

# Model, Loss, Optimizer
model = CombinedModel(resnet_config, transformer_config).to(device)
criterion = FPRPenaltyLoss(base_criterion=nn.BCELoss(), fpr_target=0.05, fpr_weight=2.0)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Metrics storage
metrics_history = {
    'train': {'losses': [], 'metrics': []},
    'val': {'losses': [], 'metrics': []}
}

# Training loop
for epoch in range(epochs):
    logger.info(f"\n{'='*20} Epoch [{epoch+1}/{epochs}] {'='*20}")
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.append(outputs.detach().cpu().numpy())
        train_labels.append(labels.detach().cpu().numpy())

    train_preds = np.concatenate(train_preds)
    train_labels = np.concatenate(train_labels)
    train_loss /= len(train_loader)
    train_metrics = calculate_metrics(train_preds, train_labels)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []

    for inputs, labels in tqdm(val_loader, desc="Validation"):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            val_loss += loss.item()
            val_preds.append(outputs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    val_loss /= len(val_loader)
    val_metrics = calculate_metrics(val_preds, val_labels)

    # Store metrics
    metrics_history['train']['losses'].append(train_loss)
    metrics_history['train']['metrics'].append(train_metrics)
    metrics_history['val']['losses'].append(val_loss)
    metrics_history['val']['metrics'].append(val_metrics)

    # Print metrics with color coding
    logger.info("\nTraining Metrics:")
    logger.info(f"  Loss: {train_loss:.4f}")
    logger.info(f"  Accuracy: {train_metrics['acc']*100:.2f}%")
    logger.info(f"  AUC-ROC: {train_metrics['auc']:.4f}")
    logger.info(f"  TPR: {train_metrics['tpr']:.4f}")
    logger.info(f"  FPR: {train_metrics['fpr']:.4f}")
    logger.info(f"  TPR@FPR≤5%: {train_metrics['tpr_at_fpr5']:.4f}")

    logger.info("\nValidation Metrics:")
    logger.info(f"  Loss: {val_loss:.4f}")
    logger.info(f"  Accuracy: {val_metrics['acc']*100:.2f}%")
    logger.info(f"  AUC-ROC: {val_metrics['auc']:.4f}")
    logger.info(f"  TPR: {val_metrics['tpr']:.4f}")
    logger.info(f"  FPR: {val_metrics['fpr']:.4f}")
    logger.info(f"  TPR@FPR≤5%: {val_metrics['tpr_at_fpr5']:.4f}")

    if val_metrics['fpr'] > 0.05:
        logger.warning(f"High FPR ({val_metrics['fpr']:.4f}) detected in validation set")
    
    # Save model after each epoch
    epoch_model_path = f'./experiments/models/{experiment_name}/eeg_model_epoch_{epoch+1:03d}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }, epoch_model_path)
    logger.info(f"Saved model checkpoint for epoch {epoch+1} to {epoch_model_path}")

# Create directories
for dir_path in [f'./experiments/{subdir}/{experiment_name}' for subdir in ['results', 'logs', 'models']]:
    os.makedirs(dir_path, exist_ok=True)

# Save the final model
final_model_path = f'./experiments/models/{experiment_name}/eeg_model_final.pth'
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_train_metrics': metrics_history['train']['metrics'][-1],
    'final_val_metrics': metrics_history['val']['metrics'][-1],
}, final_model_path)
logger.info(f"Saved final model to {final_model_path}")

# Plot and save metrics
def plot_metric(metric_name: str, train_values: list, val_values: list, ylabel: str):
    plt.figure(figsize=(12, 6))
    plt.plot(train_values, label=f'Train {metric_name}')
    plt.plot(val_values, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./experiments/results/{experiment_name}/{metric_name.lower()}.png')
    plt.close()

# Plot all metrics
metrics_to_plot = {
    'Loss': ([m['losses'][-1] for m in metrics_history['train']], [m['losses'][-1] for m in metrics_history['val']], 'Loss'),
    'Accuracy': ([m['metrics'][-1]['acc'] for m in metrics_history['train']], [m['metrics'][-1]['acc'] for m in metrics_history['val']], 'Accuracy'),
    'AUC-ROC': ([m['metrics'][-1]['auc'] for m in metrics_history['train']], [m['metrics'][-1]['auc'] for m in metrics_history['val']], 'AUC-ROC'),
    'TPR': ([m['metrics'][-1]['tpr'] for m in metrics_history['train']], [m['metrics'][-1]['tpr'] for m in metrics_history['val']], 'True Positive Rate'),
    'FPR': ([m['metrics'][-1]['fpr'] for m in metrics_history['train']], [m['metrics'][-1]['fpr'] for m in metrics_history['val']], 'False Positive Rate'),
    'TPR@FPR≤5%': ([m['metrics'][-1]['tpr_at_fpr5'] for m in metrics_history['train']], [m['metrics'][-1]['tpr_at_fpr5'] for m in metrics_history['val']], 'TPR at FPR≤5%')
}

for metric_name, (train_values, val_values, ylabel) in metrics_to_plot.items():
    plot_metric(metric_name, train_values, val_values, ylabel)

# Save detailed metrics
with open(f'./experiments/logs/{experiment_name}/metrics.txt', 'w') as f:
    f.write('epoch,train_loss,train_acc,train_auc,train_tpr,train_fpr,train_tpr_at_fpr5,'
            'val_loss,val_acc,val_auc,val_tpr,val_fpr,val_tpr_at_fpr5\n')
    for i in range(len(metrics_history['train']['losses'])):
        train_metrics = metrics_history['train']['metrics'][i]
        val_metrics = metrics_history['val']['metrics'][i]
        f.write(f"{i},"
                f"{metrics_history['train']['losses'][i]:.4f},"
                f"{train_metrics['acc']:.4f},"
                f"{train_metrics['auc']:.4f},"
                f"{train_metrics['tpr']:.4f},"
                f"{train_metrics['fpr']:.4f},"
                f"{train_metrics['tpr_at_fpr5']:.4f},"
                f"{metrics_history['val']['losses'][i]:.4f},"
                f"{val_metrics['acc']:.4f},"
                f"{val_metrics['auc']:.4f},"
                f"{val_metrics['tpr']:.4f},"
                f"{val_metrics['fpr']:.4f},"
                f"{val_metrics['tpr_at_fpr5']:.4f}\n")

logger.info("Training completed successfully!")