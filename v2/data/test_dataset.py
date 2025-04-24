import os
import h5py
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class PatientEEGTestDataset(Dataset):
    def __init__(self, data_dir, labels_json_path, window_size=20, fs=100):
        """
        Args:
            data_dir (str): Path to EEG data.
            labels_json_path (str): JSON file path (patient grouped).
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.fs = fs
        self.patients = self._load_metadata(labels_json_path)

    def _load_metadata(self, labels_json_path):
        with open(labels_json_path, "r") as f:
            metadata = json.load(f)
        return metadata["index_map"]  # List of patient records (each is a list of dicts)

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

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_records = self.patients[idx]
        patient_id = patient_records[0]["patient_id"]
        outcome = patient_records[0]["outcome"]

        eeg_segments = []

        for record in patient_records[:12]:  # Limit to 12 records
            record_name = record["record"]
            h5_path = os.path.join(self.data_dir, record_name.split("_")[0], f"{record_name}.h5")

            if not os.path.exists(h5_path):
                continue  # Skip missing files

            with h5py.File(h5_path, "r") as hf:
                eeg_data = hf["windows"][:]

            windows = self.create_windows(eeg_data)
            if windows.shape[0] == 0:
                continue  # Skip empty windows

            # For test, we can just take the first window
            eeg_tensor = torch.from_numpy(windows[0:1]).float()  # (1, C, T)
            eeg_segments.append(eeg_tensor)

        if len(eeg_segments) == 0:
            # If no valid EEG segments found, return dummy data (for safety)
            dummy_tensor = torch.zeros((1, 19, self.window_size * self.fs))
            return dummy_tensor, outcome, patient_id

        eeg_segments = torch.cat(eeg_segments, dim=0)  # (N, C, T)
        return eeg_segments, outcome, patient_id
