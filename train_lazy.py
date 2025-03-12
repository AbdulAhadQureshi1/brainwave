import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models import BiLSTMModelLazy
from torch.utils.data import DataLoader
from data.loaders.eeg_dataset_lazy import EEGDataset, create_dataloader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from utils import get_patients
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Hyperparameters
batch_size = 20
lr = 1e-2
epochs = 50
max_num_patients = None
records_per_patient = 1  # -1 for everything, otherwise specify numbers
num_windows = 10
training_dir = './training'
val_split = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
experiment_name = 'BILSTM_1WIND'

print(f'device: {device}')

# Prepare dataset
print('\nPreparing Patients')
patients_data, val_data = get_patients(training_dir, prototype=True, per_class=15, max_num_patients=max_num_patients, plot_data_distribution=True)

if len(patients_data) == 0 or len(val_data) == 0:
    print("No patient data found")
    exit()


# Model, Loss, Optimizer
print('\nLoading Model')
model = BiLSTMModelLazy()
try:
    model.load_state_dict(torch.load(f'./experiments/models/{experiment_name}/eeg_model.pth', weights_only=True))
except FileNotFoundError:
    print("\33[33m No Saved Model Found. Training From Scratch.\33[0m")

model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Metrics
train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_aucs = []
val_aucs = []
train_tprs = []
train_fprs = []
val_tprs = []
val_fprs = []

print("\nCreating DataLoader")
train_loader, val_loader = create_dataloader(patients_data, val_data, root_dir=training_dir, batch_size=batch_size, records_per_patient=records_per_patient, val_split=val_split, predict='outcome')

# Create directory if it doesn't exist
os.makedirs(f'./experiments/results/{experiment_name}', exist_ok=True)
os.makedirs(f'./experiments/logs/{experiment_name}', exist_ok=True)
os.makedirs(f'./experiments/models/{experiment_name}', exist_ok=True)

# Set Paths
results_path = f'./experiments/results/{experiment_name}'
logs_path = f'./experiments/logs/{experiment_name}'
models_path = f'./experiments/models/{experiment_name}'

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch [{epoch + 1}/{epochs}]")

    # Training phase
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []

    # print(f'Patient ID: {patient_id}')
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
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

    if len(train_preds) > 0:
        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        train_loss /= len(train_loader)

        # Calculate training metrics

        print(f'Train labels: {train_labels}')
        print(f'Train Preds: {train_preds}')

        train_acc = accuracy_score(train_labels, (train_preds > 0.5))
        train_auc = roc_auc_score(train_labels, train_preds)
    else:
        train_loss, train_acc, train_auc = 0.0, 0.0, 0.0

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []

    for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            val_loss += loss.item()
            val_preds.append(outputs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    if len(val_preds) > 0:
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_loss /= len(val_loader)

        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, (val_preds > 0.5))
        val_auc = roc_auc_score(val_labels, val_preds)
    else:
        val_loss, val_acc, val_auc = 0.0, 0.0, 0.0
    
    if len(train_preds) > 0:
        fpr_train, tpr_train, _ = roc_curve(train_labels, train_preds)
        train_fprs.append(fpr_train)
        train_tprs.append(tpr_train)

    if len(val_preds) > 0:
        fpr_val, tpr_val, _ = roc_curve(val_labels, val_preds)
        val_fprs.append(fpr_val)
        val_tprs.append(tpr_val)

    # Print Metrics
    print(f"Train Metrics:")
    print(f"  Loss: {train_loss:.4f}")
    print(f"  Accuracy: {train_acc * 100:.2f}%")
    print(f"  AUC-ROC: {train_auc:.4f}")
    # print(f"  FPR: {fpr_train} TPR: {tpr_train}")
    print(f"Val Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc * 100:.2f}%")
    print(f"  AUC-ROC: {val_auc:.4f}")
    # print(f"  FPR: {fpr_val} TPR: {tpr_val}")

    # Save metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_aucs.append(train_auc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_aucs.append(val_auc)

    torch.save(model.state_dict(), f'{models_path}/{epoch}.pth')
    with open(f'{logs_path}/metrics.txt', 'w') as f:
        if (epoch == 0):
            f.write('epoch,train_loss,train_acc,train_auc,val_loss,val_acc,val_auc\n')
        f.write(f'{epoch},{train_losses[epoch]:.4f},{train_accs[epoch]:.4f},{train_aucs[epoch]:.4f},'
                    f'{val_losses[epoch]:.4f},{val_accs[epoch]:.4f},{val_aucs[epoch]:.4f}\n')


# Save the trained model
torch.save(model.state_dict(), f'{models_path}/final.pth')

# Plot metrics
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{results_path}/loss.png')
plt.show()

plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{results_path}/accuracy.png')
plt.show()

plt.plot(train_aucs, label='Train AUC-ROC')
plt.plot(val_aucs, label='Validation AUC-ROC')
plt.xlabel('Epoch')
plt.ylabel('AUC-ROC')
plt.legend()
plt.savefig(f'{results_path}/auc.png')
plt.show()

# Save metrics in a readable format
metrics = {
    'train_losses': train_losses,
    'train_accs': train_accs,
    'train_aucs': train_aucs,
    'val_losses': val_losses,
    'val_accs': val_accs,
    'val_aucs': val_aucs
}