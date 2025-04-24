import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
import wfdb

class EEGDataset(Dataset):
    """
    PyTorch Dataset class for EEG data.
    Loads EEG data for a given patient and processes it into windows with labels.
    """

    def __init__(self, label_csv_path, records_per_patient=1, fs=100, window_size=10, predict='outcome'):
        """
        Initialize the dataset.
        
        Args:
            label_csv_path (str): Path to the CSV file containing the labels.
            records_per_patient (int): Maximum number of records to load per patient.
            fs (int): Target sampling frequency for resampling.
            window_size (int): Window size in seconds for EEG segmentation.
            predict (str): Specifies which label to predict ('outcome' or 'cpc').
        """
        self.label_csv_path = label_csv_path
        self.records_per_patient = records_per_patient
        self.fs = fs
        self.window_size = window_size
        self.predict = predict
        self.data = self._load_data()

    def _load_data(self):
        # there is no path joining anymore, just read the csv file
        data = []
        with open(self.label_csv_path, 'r') as file:
            for line in file:
                record_path, _, label = line.strip().split(',')
                try:
                    eeg_signal, sampling_rate, _, label = self.read_eeg_data(record_path)
                    if not self.is_valid_recording(eeg_signal, sampling_rate, min_duration=self.window_size):
                        print(f"Skipping record {record_path}: Duration less than {self.window_size} seconds.")
                        continue
                    eeg_signal = self.preprocess_eeg_signal(eeg_signal, sampling_rate)
                    windows, labels = self.create_windows(eeg_signal, label)
                    data.extend(zip(windows, labels))
                except Exception as e:
                    print(f"Error processing record {record_path}: {e}")
                    continue
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, label = self.data[idx]
        return torch.FloatTensor(window), torch.LongTensor([label])[0]

    def read_eeg_data(self, record_path):
        """Reads EEG data and associated metadata."""
        record = wfdb.rdrecord(record_path + '.mat')
        sampling_rate = record.fs
        channels = record.p_signal.shape[1]
        metadata_path = record_path + '.txt'
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
        # Resample to the target frequency
        eeg_signal = signal.resample(eeg_signal, int(eeg_signal.shape[0] * (self.fs / sampling_rate)), axis=0)
        # Handle NaN and Inf values
        eeg_signal = self.remove_nan_values(eeg_signal, method="interpolate")
        # Apply bandpass filtering
        eeg_signal = self.apply_filtering(eeg_signal)
        # Normalize and standardize
        eeg_signal = self.normalize_eeg_voltages(eeg_signal)
        eeg_signal = self.standardize(eeg_signal, target_channels=19)
        return eeg_signal

    def is_valid_recording(self, eeg_signal, sampling_rate, min_duration=10):
        """
        Checks if an EEG recording meets the minimum duration requirement.
        
        Args:
            eeg_signal (np.ndarray): The EEG signal array.
            sampling_rate (int): Sampling rate of the EEG signal in Hz.
            min_duration (int): Minimum duration of the recording in seconds.
        
        Returns:
            bool: True if the recording meets the minimum duration, False otherwise.
        """
        total_duration = eeg_signal.shape[0] / sampling_rate
        return total_duration >= min_duration

    def remove_nan_values(self, eeg_signal, method="zero"):
        """Removes or replaces NaN and Inf values from the EEG signal."""
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

    def apply_filtering(self, eeg_signal, lowcut=0.5, highcut=40):
        """Applies a bandpass filter to EEG data."""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(5, [low, high], btype="band")
        return signal.lfilter(b, a, eeg_signal, axis=0)

    def normalize_eeg_voltages(self, eeg_signal, norm_range=(-1, 1)):
        """Normalizes EEG voltages to a specified range."""
        min_val, max_val = norm_range
        signal_min = eeg_signal.min(axis=0, keepdims=True)
        signal_max = eeg_signal.max(axis=0, keepdims=True)
        denom = signal_max - signal_min
        denom[denom == 0] = 1
        normalized_signal = (eeg_signal - signal_min) / denom
        return normalized_signal * (max_val - min_val) + min_val

    def standardize(self, signal, target_channels=19):
        """Standardizes EEG data to a fixed number of channels."""
        if signal.shape[1] > target_channels:
            return signal[:, :target_channels]
        elif signal.shape[1] < target_channels:
            padding = np.zeros((signal.shape[0], target_channels - signal.shape[1]))
            return np.hstack((signal, padding))
        return signal

    def create_windows(self, eeg_signal, outcome):
        """Segments EEG data into fixed-size windows with labels."""
        num_time_samples = self.window_size * self.fs
        num_windows = eeg_signal.shape[0] // num_time_samples
        eeg_signal = eeg_signal[: num_windows * num_time_samples]
        eeg_windows = eeg_signal.reshape(num_windows, num_time_samples, 19)
        labels = np.full((num_windows,), outcome)
        print(f'Number of windows: {num_windows}')
        return eeg_windows, labels


def create_dataloader(patients_data, root_dir, batch_size=32, records_per_patient=1, val_split=0.2, predict='outcome'):
    """
    Create a DataLoader for training and validation datasets.

    Args:
    - patients_data (dict): Preprocessed patients' data.
    - root_dir (str): Path to the root directory containing the data.
    - batch_size (int): Batch size for DataLoader.
    - records_per_patient (int): Number of records to process per patient.
    - val_split (float): Proportion of the data to be used for validation.
    - predict (str): Specifies which label to predict ('outcome' or 'cpc').

    Returns:
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    """
    # Create the full dataset
    dataset = EEGDataset(patients_data, root_dir, window_size=20, records_per_patient=records_per_patient, predict=predict)
    
    # Calculate the sizes for training and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader