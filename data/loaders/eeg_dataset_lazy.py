import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
import wfdb
import random
import json
import time

class EEGDataset(Dataset):
    """
    PyTorch Dataset class for EEG data.
    Dynamically loads EEG data for a single patient at a time to handle large datasets.
    """

    def __init__(self, patients_data, root_dir, mode, records_per_patient=1, fs=100, window_size=10, num_windows=10, predict='outcome'):
        """
        Initialize the dataset.

        Args:
            patients_data (dict): Dictionary with patient IDs as keys and their EEG records as values.
            root_dir (str): Root directory containing the patient data.
            records_per_patient (int): Maximum number of records to load per patient.
            fs (int): Target sampling frequency for resampling.
            window_size (int): Window size in seconds for EEG segmentation.
            predict (str): Specifies which label to predict ('outcome' or 'cpc').
        """
        self.patients_data = patients_data
        self.root_dir = root_dir
        self.records_per_patient = records_per_patient
        self.fs = fs
        self.window_size = window_size
        self.num_windows = num_windows
        self.predict = predict
        self.mode = mode
        self.channels = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]

        self.index_map = self._create_index_map()

    def _create_index_map(self, cache_file="index_map.json"):
        """
        Create a mapping of global indices to patient IDs and record indices.
        Only includes records that pass the validity check.
        """
        # if not os.path.exists(cache_file):
        #     raise FileNotFoundError(f"Cache file {cache_file} does not exist.")
        
        # with open(cache_file, "r") as f:
        #     index_map = json.load(f)  # Loads as a list of lists
        
        # # Convert lists to tuples
        # index_map = [tuple(item) for item in index_map]
        # return index_map

        index_map = []
        for patient_id, records in self.patients_data.items():
            if self.records_per_patient != -1:
                # records = random.sample(records, self.records_per_patient)
                records = records[:self.records_per_patient]
            for record in records:
                if record.split("_")[-1] != "EEG":
                    continue
                record_path = os.path.join(self.root_dir, patient_id, record + '.hea')
                if self.is_valid_recording(record_path):
                    index_map.append((patient_id, record))
        return index_map
    
    def sample_indices(self, max_len):
        random_uniform = random.uniform(0,1)
        index = int(random_uniform*max_len)
        if (index + self.num_windows) > max_len:
            index = max_len - self.num_windows
        return index
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Dynamically load data for the requested index.
        """
        # starttime = time.time()
        patient_id, record = self.index_map[idx]
        record_path = os.path.join(self.root_dir, patient_id, record)
        try:
            eeg_signal, sampling_rate, channel_names, label = self.read_eeg_data(record_path, patient_id)
            # print(f'reading time: {time.time() - starttime}')
            # starttime = time.time()
            eeg_signal = self.preprocess_eeg_signal(eeg_signal, sampling_rate, channel_names)
            # print(f'preprocessing time: {time.time() - starttime}')
            # starttime = time.time()
            windows, label = self.create_windows(eeg_signal, label)
            # print(f'creating windows time: {time.time() - starttime}')
            # starttime = time.time()
            wind, _, _ = windows.shape
            if self.mode == 'train':
                start_idx= self.sample_indices(wind)
            else:
                start_idx = 0
            end_idx= start_idx+self.num_windows
            # print(f'sampling windows time: {time.time() - starttime}')
            # starttime = time.time()
            return torch.FloatTensor(windows)[start_idx:end_idx, :, :], label
        except Exception as e:
            print(f"Error processing record {record_path}: {e}")
            return None, None

    def read_eeg_data(self, record_name, patient_id):
        """Reads EEG data and associated metadata."""
        record = wfdb.rdrecord(record_name)
        sampling_rate = record.fs
        channels = record.p_signal.shape[1]
        channel_names = record.sig_name 

        metadata_path = os.path.join(self.root_dir, patient_id, f"{patient_id}.txt")
        label_mapping = {"Good": 0, "Poor": 1}

        with open(metadata_path, "r") as file:
            metadata = file.readlines()
            # Choose the label based on the 'predict' parameter
            if self.predict == "outcome":
                label = [line.split(":")[-1].strip() for line in metadata if "Outcome" in line][0]
                label = label_mapping.get(label, -1)  # Default to -1 if outcome is unrecognized
            elif self.predict == "cpc":
                label = [int(line.split(":")[-1].strip()) for line in metadata if "CPC" in line][0]
            else:
                raise ValueError("Invalid 'predict' value. Must be 'outcome' or 'cpc'.")
        
        return record.p_signal, sampling_rate, channel_names, label

    def preprocess_eeg_signal(self, eeg_signal, sampling_rate, channel_names):
        """Applies preprocessing steps to the EEG signal."""
        # Resample to the target frequency
        # Trim to the first 40 minutes if necessary
        # eeg_signal = self.limit_recording_duration(eeg_signal, sampling_rate, max_duration=40 * 60)
        # Handle NaN and Inf values
        eeg_signal = self.remove_nan_values(eeg_signal, method="interpolate")
        # Resample 
        eeg_signal = signal.resample(eeg_signal, int(eeg_signal.shape[0] * (self.fs / sampling_rate)), axis=0)
        # Apply bandpass filtering
        eeg_signal = self.apply_filtering(eeg_signal)
        # Normalize and standardize
        eeg_signal = self.normalize_eeg_voltages(eeg_signal)
        eeg_signal = self.standardize(eeg_signal, channel_names)
        return eeg_signal

    def limit_recording_duration(self, eeg_signal, sampling_rate, max_duration=2400):
        """
        Limits the EEG signal to the first `max_duration` seconds.
        
        Args:
            eeg_signal (np.ndarray): The EEG signal array.
            sampling_rate (int): Sampling rate of the EEG signal in Hz.
            max_duration (int): Maximum duration of the recording in seconds (default: 40 minutes).
        
        Returns:
            np.ndarray: Trimmed EEG signal.
        """
        max_samples = int(max_duration * sampling_rate)
        if eeg_signal.shape[0] > max_samples:
            eeg_signal = eeg_signal[:max_samples, :]
        return eeg_signal

    def is_valid_recording(self, header_path, min_duration=610):
        """
        Check if the recording meets the minimum duration requirement using the header file.
        
        Args:
            header_path (str): Path to the .hea header file.
            min_duration (int): Minimum duration of the recording in seconds.
        
        Returns:
            bool: True if the recording meets the minimum duration, False otherwise.
        """
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

    def standardize(self, eeg_signal, channel_names):
        """ 
        Standardizes EEG data to match a fixed set of channels in a consistent order.
        
        Args:
            eeg_signal (np.ndarray): EEG data array (samples, channels).
            channel_names (list): List of channel names corresponding to eeg_signal columns.
        
        Returns:
            np.ndarray: Standardized EEG signal with channels ordered as in self.channels.
        """
        target_channels = self.channels  # Predefined consistent channel order
        num_samples = eeg_signal.shape[0]

        # Create a mapping of channel names to indices in the input signal
        channel_index_map = {name: i for i, name in enumerate(channel_names)}

        # Initialize an array with zeros
        standardized_signal = np.zeros((num_samples, len(target_channels)))

        # Fill standardized_signal with matching channels
        for i, channel in enumerate(target_channels):
            if channel in channel_index_map:
                standardized_signal[:, i] = eeg_signal[:, channel_index_map[channel]]  # Assign existing channel
            else:
                standardized_signal[:, i] = 0  # Assign zeros for missing channels

        return standardized_signal

    def create_windows(self, eeg_signal, outcome):
        """Segments EEG data into fixed-size windows with labels."""
        num_time_samples = self.window_size * self.fs
        num_windows = eeg_signal.shape[0] // num_time_samples
        eeg_signal = eeg_signal[: num_windows * num_time_samples]
        eeg_windows = eeg_signal.reshape(num_windows, num_time_samples, 19)
        # labels = np.full((num_windows,), outcome)
        # print(f'Number of windows: {num_windows}')
        # print(f'outcome: {outcome}')
        return eeg_windows, outcome


def create_dataloader(patients_data, val_data, root_dir, batch_size=32, records_per_patient=-1, val_split=0.2, predict='outcome'):
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
    train_dataset = EEGDataset(patients_data, root_dir, mode='train', records_per_patient=records_per_patient, predict=predict)
    val_dataset = EEGDataset(val_data, root_dir, mode='validation', records_per_patient=records_per_patient, predict=predict)

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=8)

    return train_loader, val_loader