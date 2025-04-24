
import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
# import csv
import sys
import random
import h5py
import json

class EEGDataset(Dataset):
    def __init__(self, data_dir, labels_json_path, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing preprocessed EEG data.
            label_csv_path (str): Path to the CSV file containing record names and labels.
            transform (callable, optional): Optional transform to be applied to the EEG signal.
        """
        self.transform = transform
        self.window_size = 20
        self.fs = 100
        self.records, self.labels = self._load_labels(labels_json_path)
        self.root_dir = data_dir
        
    def _load_labels(self, labels_json_path):
        """Load record names and labels from the JSON file."""
        records = []
        labels = []

        with open(labels_json_path, "r") as json_file:
            metadata = json.load(json_file)

        for entry in metadata["index_map"]:
            records.append(entry["record"])
            labels.append(entry["outcome"])
        
        return records, labels
    
    def create_windows(self, eeg_signal):
        num_time_samples = self.window_size * self.fs
        num_windows = eeg_signal.shape[0] // num_time_samples
        eeg_signal = eeg_signal[:, :num_windows * num_time_samples]
        eeg_windows = eeg_signal.reshape(
            eeg_signal.shape[1],
            num_windows,
            num_time_samples
        ).transpose(1, 0, 2)
        
        return eeg_windows
    
    def sample_indices(self, max_len):
        random_uniform = random.uniform(0,1)
        index = int(random_uniform*max_len)
        if (index + 1) > max_len:
            index = max_len - 1
        return index

    def __len__(self):
        """Return the total number of samples."""
        return len(self.records)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            eeg_signal (torch.Tensor): The preprocessed EEG signal of shape (channels, time_samples).
            label (int): The corresponding label.
        """
        record_name = self.records[idx]
        label = self.labels[idx]
        h5_path = self.root_dir + "/" + record_name.split('_')[0] + "/" + record_name + '.h5'

        with h5py.File(h5_path, "r") as hf:
            eeg_signal = hf["windows"][:]

        windows = self.create_windows(eeg_signal)
        wind, _, _ = windows.shape
        start_idx = self.sample_indices(wind)
        end_idx = start_idx + 1
        
        eeg_signal = windows[start_idx:end_idx]
        eeg_signal = torch.from_numpy(eeg_signal).float()
        
        if self.transform:
            eeg_signal = self.transform(eeg_signal)
        
        return eeg_signal, label
