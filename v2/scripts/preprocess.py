import os
import json
import numpy as np
from tqdm import tqdm
import wfdb
from scipy import signal
import h5py  # New import for HDF5 support

def read_eeg_data(record_name, patient_id, root_dir, predict):
    """Reads EEG data and associated metadata."""
    record = wfdb.rdrecord(record_name)
    sampling_rate = record.fs
    channels = record.p_signal.shape[1]

    metadata_path = os.path.join(root_dir, patient_id, f"{patient_id}.txt")
    label_mapping = {"Good": 1, "Poor": 0}

    with open(metadata_path, "r") as file:
        metadata = file.readlines()
        if predict == "outcome":
            label = [line.split(":")[-1].strip() for line in metadata if "Outcome" in line][0]
            label = label_mapping.get(label, -1)  # Default to -1 if outcome is unrecognized
        elif predict == "cpc":
            label = [int(line.split(":")[-1].strip()) for line in metadata if "CPC" in line][0]
        else:
            raise ValueError("Invalid 'predict' value. Must be 'outcome' or 'cpc'.")
    
    return record.p_signal, sampling_rate, channels, label

def preprocess_eeg_signal(eeg_signal, sampling_rate, target_fs):
    """Applies preprocessing steps to the EEG signal."""
    eeg_signal = limit_recording_duration(eeg_signal, sampling_rate, max_duration=40 * 60)
    eeg_signal = signal.resample(eeg_signal, int(eeg_signal.shape[0] * (target_fs / sampling_rate)), axis=0)
    eeg_signal = remove_nan_values(eeg_signal, method="interpolate")
    eeg_signal = apply_filtering(eeg_signal, target_fs)
    eeg_signal = normalize_eeg_voltages(eeg_signal)
    eeg_signal = standardize(eeg_signal, target_channels=19)
    return eeg_signal

def limit_recording_duration(eeg_signal, sampling_rate, max_duration=2400):
    """Limits the EEG signal to the first `max_duration` seconds."""
    max_samples = int(max_duration * sampling_rate)
    return eeg_signal[:max_samples, :] if eeg_signal.shape[0] > max_samples else eeg_signal

def is_valid_recording(header_path, min_duration=610):
    """Check if the recording meets the minimum duration requirement using the header file."""
    try:
        with open(header_path, "r") as f:
            lines = f.readlines()
        first_line = lines[0].strip().split()
        num_samples = int(first_line[3])
        sampling_frequency = int(first_line[2])
        return (num_samples / sampling_frequency) >= min_duration
    except Exception as e:
        print(f"Error reading header file {header_path}: {e}")
        return False

def remove_nan_values(eeg_signal, method="zero"):
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

def apply_filtering(eeg_signal, fs, lowcut=0.5, highcut=40):
    """Applies a bandpass filter to EEG data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(5, [low, high], btype="band")
    return signal.lfilter(b, a, eeg_signal, axis=0)

def normalize_eeg_voltages(eeg_signal, norm_range=(-1, 1)):
    """Normalizes EEG voltages to a specified range."""
    min_val, max_val = norm_range
    signal_min = eeg_signal.min(axis=0, keepdims=True)
    signal_max = eeg_signal.max(axis=0, keepdims=True)
    denom = signal_max - signal_min
    denom[denom == 0] = 1
    normalized_signal = (eeg_signal - signal_min) / denom
    return normalized_signal * (max_val - min_val) + min_val

def standardize(eeg_signal, target_channels=19):
    """Standardizes EEG data to a fixed number of channels."""
    if eeg_signal.shape[1] > target_channels:
        return eeg_signal[:, :target_channels]
    elif eeg_signal.shape[1] < target_channels:
        padding = np.zeros((eeg_signal.shape[0], target_channels - eeg_signal.shape[1]))
        return np.hstack((eeg_signal, padding))
    return eeg_signal

def create_windows(eeg_signal, window_size, fs):
    """Segments EEG data into fixed-size windows."""
    num_time_samples = window_size * fs
    num_windows = eeg_signal.shape[0] // num_time_samples
    eeg_signal = eeg_signal[: num_windows * num_time_samples]
    # Ensure there are 19 channels in the reshaped data
    eeg_windows = eeg_signal.reshape(num_windows, num_time_samples, 19)
    return eeg_windows[0]

def preprocess_and_cache(root_dir, cache_dir, metadata_path, fs=100, window_size=10, min_duration=610, predict="cpc"):
    metadata = {'index_map': []}
    os.makedirs(cache_dir, exist_ok=True)
    
    for patient_id in tqdm(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        
        # Load patient metadata
        metadata_path = os.path.join(patient_dir, f"{patient_id}.txt")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            lines = f.readlines()
            outcome = next(line.split(":")[-1].strip() for line in lines if "Outcome" in line)
            cpc = int(next(line.split(":")[-1].strip() for line in lines if "CPC" in line))
        
        # Process each record
        for record in os.listdir(patient_dir):
            if not record.endswith('.hea'):
                continue
            if not "_EEG" in record:
                continue
            record_base = record[:-4]
            record_path = os.path.join(patient_dir, record_base)
            
            # Change cache file extension to .h5 for HDF5 format
            cache_path = os.path.join(cache_dir, patient_id, f"{record_base}.h5")
            if os.path.exists(cache_path):
                continue
            
            # Validate and preprocess
            if not is_valid_recording(record_path + '.hea', min_duration):
                continue
            
            try:
                # Load and preprocess data
                eeg_signal, sampling_rate, _, _ = read_eeg_data(record_path, patient_id, root_dir, predict)
                eeg_signal = preprocess_eeg_signal(eeg_signal, sampling_rate, fs)
                windows = create_windows(eeg_signal, window_size, fs)
                
                # Save preprocessed windows in HDF5 format for faster I/O
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with h5py.File(cache_path, "w") as hf:
                    hf.create_dataset("windows", data=windows, compression="gzip", compression_opts=4)
                
                # Add to metadata   
                metadata['index_map'].append({
                    'patient_id': patient_id,
                    'record': record_base,
                    'num_windows': windows.shape[0],
                    'outcome': 1 if outcome == "Good" else 0,
                    'cpc': cpc
                })
            except Exception as e:
                print(f"Skipping {record_path}: {str(e)}")
                
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

preprocess_and_cache(
    root_dir='/media/brainwave/2Tb HDD/physionet.org/files/i-care/2.1/training',
    cache_dir='../preprocessed_data',
    metadata_path='../metadata.json',
    fs=100,
    window_size=180,
    min_duration=610,
    predict="cpc"
)