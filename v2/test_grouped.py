from torch.utils.data import DataLoader
from data.test_dataset import PatientEEGTestDataset  
from models.model import CombinedModel
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch
import numpy as np
from tqdm import tqdm

# Load model
def load_model(model_path, device):
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
        'verbose': False
    }
    transformer_config = {
        'output_dim': 1,
        'hidden_dim': 32,
        'num_heads': 4,
        'key_query_dim': 32,
        'intermediate_dim': 128
    }
    model = CombinedModel(resnet_config, transformer_config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './experiments/hybrid_model_nblocks4_kernel17_basef64_transformerhidden32_20secwindow_weightedloss_13/best_model.pth'
model = load_model(model_path, device)

# Load test dataset
test_dataset = PatientEEGTestDataset(
    data_dir="./preprocessed_data",
    labels_json_path="./test_metadata_grouped.json"
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Inference
true_labels = []
pred_labels = []
avg_probs_list = []

with torch.no_grad():
    for eeg_segments, label, patient_id in test_loader:
        eeg_segments = eeg_segments.squeeze(0).to(device)  # (N, C, T)

        if eeg_segments.ndim < 3 or eeg_segments.shape[0] == 0:
            continue  # Skip invalid

        outputs = model(eeg_segments)  # (N, 1)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        threshold = 0.65

        # Aggregate
        avg_prob = np.mean(probs)
        majority_vote = (probs > threshold).astype(int).mean() > threshold

        avg_probs_list.append(avg_prob)
        # pred_labels.append(int(avg_prob > threshold))  # Binary prediction
        pred_labels.append(int(majority_vote))
        true_labels.append(label[0])
        print(f"True: {label[0]}, Pred: {avg_prob > threshold}")

        print(f'PID: {patient_id[0]} Avg: {avg_prob} Majority: {int(majority_vote)}')

# Evaluation
acc = accuracy_score(true_labels, pred_labels)
auc = roc_auc_score(true_labels, avg_probs_list)
tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
cs = tp / (fp + fn)

print(f"\n--- Patient-Level Evaluation ---")
print(f"Accuracy: {acc*100:.2f}%")
print(f"AUC: {auc:.4f}")
print(f"TPR: {tpr:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"Challenge Score at 12hrs: {cs:.4f}")
