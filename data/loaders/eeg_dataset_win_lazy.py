import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
import wfdb
import random
import mne
mne.set_log_level(verbose='WARNING')
from mne.filter import filter_data, notch_filter


class EEGDatasetWinLazy(Dataset):
    def __init__(self, patients_data, root_dir, records_per_patient=1, fs=100, window_size=10, predict='outcome'):
        self.patients_data = patients_data
        self.root_dir = root_dir
        self.records_per_patient = records_per_patient
        self.fs = fs
        self.window_size = window_size
        self.predict = predict
        self.data_index = self._create_index()
    
    def _create_index(self):
        index = []
        for patient_id, records in self.patients_data.items():
            if self.records_per_patient != -1:
                records = records[:self.records_per_patient]
            for record in records:
                record = os.path.join(self.root_dir, patient_id, record)
                record_path = record + '.hea'
                if self.is_valid_recording(record_path):
                    index.append((patient_id, record))
        outcomes = [0, 0]
        for p_tuple in index:
            pid, rec = p_tuple
            metadata_path = os.path.join(self.root_dir, pid, f"{pid}.txt")
            with open(metadata_path, 'r') as file:
                metadata = file.readlines()
                label = [line.split(":")[-1].strip() for line in metadata if "Outcome" in line][0]
                label_mapping = {"Good": 0, "Poor": 1}
                label = label_mapping.get(label)  # Default to -1 if outcome is unrecognized  
                outcomes[label] += 1          
        print(f'Good: {outcomes[0]} Poor:{outcomes[1]}')
        return index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        patient_id, record_path = self.data_index[idx]
        eeg_signal, sampling_rate, _, label = self.read_eeg_data(record_path, patient_id)
        eeg_signal = self.preprocess_eeg_signal(eeg_signal, sampling_rate)
        windows, _ = self.create_windows(eeg_signal, label)
        wind, _, _ = windows.shape
        start_idx = self.sample_indices(wind)
        end_idx = start_idx + 1
        return torch.FloatTensor(windows)[start_idx:end_idx, :, :], label

    def sample_indices(self, max_len):
        random_uniform = random.uniform(0,1)
        index = int(random_uniform*max_len)
        if (index + 1) > max_len:
            index = max_len - 1
        return index

    def read_eeg_data(self, record_name, patient_id):
        record = wfdb.rdrecord(record_name)
        sampling_rate = record.fs
        channels = record.p_signal.shape[1]

        metadata_path = os.path.join(self.root_dir, patient_id, f"{patient_id}.txt")
        label_mapping = {"Good": 0, "Poor": 1}

        with open(metadata_path, "r") as file:
            metadata = file.readlines()
            if self.predict == "outcome":
                label = [line.split(":")[-1].strip() for line in metadata if "Outcome" in line][0]
                label = label_mapping.get(label, -1)  # Default to -1 if outcome is unrecognized
            elif self.predict == "cpc":
                label = [int(line.split(":")[-1].strip()) for line in metadata if "CPC" in line][0]
            else:
                raise ValueError("Invalid 'predict' value. Must be 'outcome' or 'cpc'.")
        
        return record.p_signal, sampling_rate, channels, label

    def preprocess_eeg_signal(self, eeg_signal, sampling_rate):
        """Applies preprocessing steps to the EEG signal."""
        eeg_signal = self.remove_nan_values(eeg_signal, method="interpolate")
        # Resample 
        eeg_signal = signal.resample(eeg_signal, int(eeg_signal.shape[0] * (self.fs / sampling_rate)), axis=0)
        # Apply bandpass filtering
        eeg_signal = self.apply_filtering(eeg_signal)
        # Normalize and standardize
        eeg_signal = self.normalize_eeg_voltages(eeg_signal)
        eeg_signal = self.standardize(eeg_signal)
        return eeg_signal
    
    def apply_filtering(self, eeg_signal, lowcut=0.5, highcut=40):
        """Applies a bandpass filter to EEG data."""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(5, [low, high], btype="band")
        return signal.lfilter(b, a, eeg_signal, axis=0)

    def remove_nan_values(self, eeg_signal, method="interpolate"):
        eeg_signal[np.isinf(eeg_signal)] = np.nan

        if method == "zero":
            return np.nan_to_num(eeg_signal, nan=0.0)
        elif method == "mean":
            channel_means = np.nanmean(eeg_signal, axis=0)
            channel_means = np.where(np.isnan(channel_means), 0, channel_means)
            return np.where(np.isnan(eeg_signal), channel_means, eeg_signal)
        elif method == "interpolate":
            cleaned_signal = np.copy(eeg_signal)
            for channel in range(eeg_signal.shape[1]):
                nan_indices = np.isnan(eeg_signal[:, channel])
                if np.all(nan_indices):
                    cleaned_signal[:, channel] = 0.0
                else:
                    x = np.arange(eeg_signal.shape[0])
                    valid_x = x[~nan_indices]
                    valid_y = eeg_signal[~nan_indices, channel]
                    cleaned_signal[:, channel] = np.interp(x, valid_x, valid_y)
            return cleaned_signal
        else:
            raise ValueError(f"Unsupported method: {method}")

    def normalize_eeg_voltages(self, eeg_signal, norm_range=(-1, 1)):
        min_val, max_val = norm_range
        signal_min = eeg_signal.min(axis=0, keepdims=True)
        signal_max = eeg_signal.max(axis=0, keepdims=True)
        denom = signal_max - signal_min
        denom[denom == 0] = 1
        normalized_signal = (eeg_signal - signal_min) / denom
        return normalized_signal * (max_val - min_val) + min_val

    def standardize(self, signal, target_channels=19):
        if signal.shape[1] > target_channels:
            return signal[:, :target_channels]
        elif signal.shape[1] < target_channels:
            padding = np.zeros((signal.shape[0], target_channels - signal.shape[1]))
            return np.hstack((signal, padding))
        return signal

    def is_valid_recording(self, header_path, min_duration=20):
        try:
            with open(header_path, "r") as f:
                lines = f.readlines()
            
            # Parse the first line of the header file
            first_line = lines[0].strip().split()
            num_samples = int(first_line[3])  # Number of samples
            sampling_frequency = int(first_line[2])  # Sampling frequency in Hz
            
            # Calculate duration
            duration = num_samples / sampling_frequency
            return duration >= min_duration
        except Exception as e:
            print(f"Error reading header file {header_path}: {e}")
            return False

    def create_windows(self, eeg_signal, outcome):
        num_time_samples = self.window_size * self.fs
        num_windows = eeg_signal.shape[0] // num_time_samples
        eeg_signal = eeg_signal[: num_windows * num_time_samples]
        eeg_windows = eeg_signal.reshape(num_windows, num_time_samples, 19)
        labels = np.full((num_windows,), outcome)
        return eeg_windows, labels


def create_dataloader(patients_data, root_dir, batch_size=32, records_per_patient=1, val_split=0.2, predict='outcome'):
    dataset = EEGDatasetWinLazy(patients_data, root_dir, window_size=20, records_per_patient=records_per_patient, predict=predict)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=6, pin_memory=True)
    
    return train_loader, val_loader

def create_test_loader(patients_data, root_dir, batch_size=32, records_per_patient=1, predict='outcome'):
    dataset = EEGDatasetWinLazy(patients_data, root_dir, window_size=20, records_per_patient=records_per_patient, predict=predict)
    test_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=6, pin_memory=True)
    return test_loader 