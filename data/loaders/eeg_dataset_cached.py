import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
import wfdb
import random
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import h5py
import hashlib


class EEGDatasetCached(Dataset):
    def __init__(self, cache_file):
        """
        Initialize the dataset with a pre-cached HDF5 file
        Args:
            cache_file: Path to the HDF5 cache file containing preprocessed EEG data
        """
        self.cache_file = cache_file
        
        # Count total samples and verify cache file
        with h5py.File(self.cache_file, 'r') as f:
            self.num_samples = len(f.keys())
            if self.num_samples == 0:
                raise ValueError("Cache file is empty")
            
            # Count outcomes for reporting class distribution
            outcomes = [0, 0]
            for idx in range(self.num_samples):
                label = f[str(idx)]['label'][()]
                outcomes[label] += 1
            print(f'Dataset distribution - Good: {outcomes[0]} Poor: {outcomes[1]}')
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.cache_file, 'r') as f:
            grp = f[str(idx)]
            windows = grp['windows'][:]
            label = grp['label'][()]
            
            # Sample a random window
            wind, _, _ = windows.shape
            start_idx = self.sample_indices(wind)
            end_idx = start_idx + 1
            
            return torch.FloatTensor(windows[start_idx:end_idx, :, :]), label

    @lru_cache(maxsize=1000)
    def sample_indices(self, max_len):
        """Cache sample indices for frequently accessed lengths"""
        random_uniform = random.uniform(0, 1)
        index = int(random_uniform * max_len)
        if (index + 1) > max_len:
            index = max_len - 1
        return index


def create_dataloader(cache_file, batch_size=32, val_split=0.2, num_workers=None):
    """
    Create train and validation dataloaders from cached data
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Use at most 8 workers
        
    dataset = EEGDatasetCached(cache_file)
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader


def create_test_loader(cache_file, batch_size=32, num_workers=None):
    """
    Create test dataloader from cached data
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
        
    dataset = EEGDatasetCached(cache_file)

    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return test_loader