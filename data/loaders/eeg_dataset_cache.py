import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
import wfdb
import json

class EEGDataset(Dataset):
    def __init__(self, metadata_path, cache_dir, num_windows=30, predict='outcome'):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.index_map = [
            entry for entry in metadata['index_map']
            if entry['num_windows'] >= num_windows
        ]
        
        print(self.index_map)
        
        self.cache_dir = cache_dir
        self.num_windows = num_windows
        self.predict = predict

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        entry = self.index_map[idx]
        cache_path = os.path.join(
            self.cache_dir,
            entry['patient_id'],
            f"{entry['record']}.npy"
        )
        windows = np.load(cache_path)
        
        # Randomly sample consecutive windows
        max_start = windows.shape[0] - self.num_windows
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        selected = windows[start_idx:start_idx + self.num_windows]
        
        label = entry[self.predict]
        return torch.FloatTensor(selected), torch.tensor(label, dtype=torch.long)

def create_dataloader(metadata_path, cache_dir, batch_size=32, num_windows=30, predict='outcome'):
    dataset = EEGDataset(metadata_path, cache_dir, num_windows, predict)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )