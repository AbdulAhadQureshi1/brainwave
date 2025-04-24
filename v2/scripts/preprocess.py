# import os
# import shutil
# import scipy.io
# from scipy import signal
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import csv

# class EEGPreprocessor:
#     def __init__(self, input_dir, output_dir, prev_dir, target_fs=100, skip_processed=False, records_per_patient=None,
#                 filter_low=0.5, filter_high=40.0, filter_order=3):
#         self.input_dir = input_dir
#         self.output_dir = output_dir
#         self.prev_dir = prev_dir
#         self.target_fs = target_fs
#         self.records_per_patient = records_per_patient
#         self.skip_processed = skip_processed
#         self.filter_low = filter_low
#         self.filter_high = filter_high
#         self.filter_order = filter_order
#         self.target_channels = 19

#     def process_dataset(self):
#         """Process all patient directories and save processed data"""
#         # Create root output directory
#         completed_patients = []
#         try:
#             completed_patients.extend(os.listdir(self.prev_dir))
#             # print(completed_patients)
#             # print("nigga not")
#             # sys.exit(420)
#         except IOError as e:
#             print("nigga what", e)
#             sys.exit(69)

#         os.makedirs(self.output_dir, exist_ok=True)

#         # Process each patient directory
#         for patient_id in os.listdir(self.input_dir):
#             if patient_id in completed_patients:
#                 continue
#             patient_src = os.path.join(self.input_dir, patient_id)
#             if not os.path.isdir(patient_src):
#                 continue
                
#             patient_dst = os.path.join(self.output_dir, patient_id)
#             os.makedirs(patient_dst, exist_ok=True)
            
#             self._process_patient_directory(patient_src, patient_dst)

#         # Copy root files (index.html and RECORDS)
#         self._copy_root_files()

#     def _process_patient_directory(self, src_dir, dst_dir):
#         """Process all records in a patient directory"""
#         # Filter files ending with "_EEG.mat"
#         mat_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_EEG.mat')])
        
#         # Apply records per patient limit
#         if self.records_per_patient:
#             mat_files = mat_files[:self.records_per_patient]

#         for mat_file in mat_files:
#             base_name = mat_file[:-4]  # Remove .mat extension
#             hea_file = base_name + '.hea'
#             print(f"Processing {base_name}")
#             # Process EEG data
#             self._process_eeg_file(src_dir, dst_dir, base_name)
            
#         # Copy non-EEG files
#         self._copy_non_eeg_files(src_dir, dst_dir)

#     def _process_eeg_file(self, src_dir, dst_dir, base_name):
#         """Process a single EEG record"""
#         mat_src = os.path.join(src_dir, base_name + '.mat')
#         hea_src = os.path.join(src_dir, base_name + '.hea')

#         try:
#             # Load original data
#             original_signal = scipy.io.loadmat(mat_src)['val']    
#             original_fs = self._read_sampling_rate(hea_src)
            
#             # Preprocess signal
#             processed_signal = self._preprocess(original_signal, original_fs)
            
#             # Save processed data
#             self._save_processed_data(processed_signal, dst_dir, base_name)
#             self._update_header_file(hea_src, dst_dir, base_name, original_fs)
            
#         except Exception as e:
#             print(f"Error processing {base_name}: {str(e)}")

#     def _preprocess(self, signal, original_fs):
#         """Apply all preprocessing steps"""
#         # Adjust number of channels to 19
#         signal = self._adjust_channels(signal)
        
#         # Resample
#         if original_fs != self.target_fs:
#             signal = self._resample_signal(signal, original_fs)
        
#         # Remove NaNs
#         signal = self._handle_nans(signal)
        
#         # Filter
#         signal = self._apply_bandpass_filter(signal)
        
#         # Normalize
#         signal = self._normalize_signal(signal)
        
#         return signal

#     def _adjust_channels(self, signal):
#         """
#         Adjust the number of channels to 19.
#         - If the signal has more than 19 channels, truncate it.
#         - If the signal has fewer than 19 channels, pad it with zeros.
#         """
#         num_channels = signal.shape[0]  # Shape is (channels, time_samples)
        
#         if num_channels > self.target_channels:
#             # Truncate to 19 channels
#             signal = signal[:self.target_channels, :]
#         elif num_channels < self.target_channels:
#             # Pad with zeros to 19 channels
#             padding = np.zeros((self.target_channels - num_channels, signal.shape[1]))
#             signal = np.vstack((signal, padding))
        
#         return signal

#     def _resample_signal(self, signal, original_fs):
#         """Resample to target frequency using scipy.signal.resample"""
#         num_samples = int(signal.shape[1] * (self.target_fs / original_fs))  # Resample time_samples axis
#         resampled_signal = np.zeros((signal.shape[0], num_samples))  # Initialize output array
        
#         # Resample each channel
#         for i in range(signal.shape[0]):
#             resampled_signal[i, :] = scipy.signal.resample(signal[i, :], num_samples)
        
#         return resampled_signal

#     def _handle_nans(self, signal):
#         """Interpolate NaN values"""
#         for i in range(signal.shape[0]):  # Iterate over channels
#             mask = np.isnan(signal[i, :])
#             signal[i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), signal[i, ~mask])
#         return signal

#     def _apply_bandpass_filter(self, signal):
#         """Apply Butterworth bandpass filter"""
#         nyq = 0.5 * self.target_fs
#         low = self.filter_low / nyq
#         high = self.filter_high / nyq
#         b, a = scipy.signal.butter(self.filter_order, [low, high], btype='band')
        
#         return scipy.signal.lfilter(b, a, signal, axis=0)

#     def _normalize_signal(self, signal):
#         """Normalize signal to zero mean and unit variance"""
#         epsilon = 1e-10  # Small constant to avoid division by zero
#         mean = np.mean(signal, axis=1, keepdims=True)  # Mean over time_samples axis
#         std = np.std(signal, axis=1, keepdims=True)  # Std over time_samples axis
        
#         # Handle zero standard deviation
#         std[std == 0] = epsilon
        
#         # Normalize
#         normalized_signal = (signal - mean) / std
        
#         # Handle NaN values (if any)
#         if np.isnan(normalized_signal).any():
#             print("Warning: NaN values detected in normalized signal. Replacing with zeros.")
#             normalized_signal = np.nan_to_num(normalized_signal)
        
#         return normalized_signal

#     def _read_sampling_rate(self, hea_path):
#         """Read sampling rate from header file"""
#         with open(hea_path, 'r') as f:
#             header = f.readline().strip().split()
#         return float(header[2])

#     def _save_processed_data(self, signal, dst_dir, base_name):
#         """Save processed signal with compression"""
#         output_path = os.path.join(dst_dir, base_name + '.mat')
#         scipy.io.savemat(output_path, {'eeg_signal': signal}, 
#                         do_compression=True, format='5')

#     def _update_header_file(self, hea_src, dst_dir, base_name, original_fs):
#         """Update header file with new sampling rate"""
#         # Read original header
#         with open(hea_src, 'r') as f:
#             lines = f.readlines()
        
#         # Update sampling rate information
#         header = lines[0].strip().split()
#         header[2] = str(self.target_fs)
#         if len(header) >= 6:
#             header[5] = str(self.target_fs)
#         lines[0] = ' '.join(header) + '\n'
        
#         # Write updated header
#         hea_dst = os.path.join(dst_dir, base_name + '.hea')
#         with open(hea_dst, 'w') as f:
#             f.writelines(lines)

#     def _copy_non_eeg_files(self, src_dir, dst_dir):
#         """Copy all non-EEG files"""
#         for f in os.listdir(src_dir):
#             if not f.endswith(('.mat', '.hea')):
#                 src = os.path.join(src_dir, f)
#                 dst = os.path.join(dst_dir, f)
#                 if os.path.isfile(src):
#                     shutil.copy2(src, dst)

#     def _copy_root_files(self):
#         """Copy root directory files"""
#         for f in ['index.html', 'RECORDS']:
#             src = os.path.join(self.input_dir, f)
#             if os.path.exists(src):
#                 shutil.copy2(src, self.output_dir)

#     def generate_record_label_csv(self, output_csv_path):
#         """
#         Loop over the original training folder and create a CSV file of record_name and label.
        
#         Args:
#             output_csv_path (str): Path to save the CSV file.
#         """
#         records = []
#         for patient_id in os.listdir(self.input_dir):
#             patient_dir = os.path.join(self.input_dir, patient_id)
#             if not os.path.isdir(patient_dir):
#                 continue
            
#             for mat_file in os.listdir(patient_dir):
#                 if mat_file.endswith('_EEG.mat'):
#                     base_name = mat_file[:-4]
#                     hea_file = os.path.join(patient_dir, base_name + '.hea')
#                     outcome_map = {"Good": 0, "Poor": 1}
#                     try:
#                         metadata_path = os.path.join(self.input_dir, patient_id, patient_id + '.txt')
#                         with open(metadata_path, 'r') as f:
#                             metadata = f.readlines()
#                             outcome = [line.split(":")[-1].strip() for line in metadata if "Outcome" in line][0]
#                             outcome = outcome_map.get(outcome)
#                             cpc = [int(line.split(":")[-1].strip()) for line in metadata if "CPC" in line][0]
                            
#                         records.append((base_name, cpc, outcome))
#                     except Exception as e:
#                         print(f"Error reading label for {base_name}: {str(e)}")
        
#         with open(output_csv_path, mode='w', newline='') as csv_file:
#             writer = csv.writer(csv_file)
#             writer.writerow(['record_name', 'cpc', 'outcome'])
#             writer.writerows(records)
        
#         print(f"CSV file saved to {output_csv_path}")


#     def generate_output_csv(self, output_csv_name):
#         eeg_files = []
#         records = []
#         eeg_files.extend([os.path.join(self.output_dir, i) for i in os.listdir(self.output_dir)])
#         eeg_files.extend([os.path.join(self.prev_dir, i) for i in os.listdir(self.prev_dir)])
#         eeg_files = [f for f in eeg_files if os.path.isdir(f)]
#         eeg_files = [os.path.join(eeg_file, f) for eeg_file in eeg_files for f in os.listdir(eeg_file) if f.endswith('_EEG.mat')]

#         for eeg_file in eeg_files:
#             base_name = eeg_file[:-4]
#             outcome_map = {"Good": 0, "Poor": 1}
#             try:
#                 metadata_path = os.path.join(os.path.dirname(eeg_file), os.path.basename(os.path.dirname(eeg_file)) + '.txt')
#                 with open(metadata_path, 'r') as f:
#                     metadata = f.readlines()
#                     outcome = [line.split(":")[-1].strip() for line in metadata if "Outcome" in line][0]
#                     outcome = outcome_map.get(outcome)
#                     cpc = [int(line.split(":")[-1].strip()) for line in metadata if "CPC" in line][0]
                    
#                 records.append((base_name, cpc, outcome))
#             except Exception as e:
#                 print(f"Error reading label for {base_name}: {str(e)}")

#         with open(output_csv_name, mode='w', newline='') as csv_file:
#             writer = csv.writer(csv_file)
#             writer.writerow(['record_name', 'cpc', 'outcome'])
#             writer.writerows(records)

#     def validate_dataset(self):
#         pids = [id for id in os.listdir(self.input_dir)]
#         invalid_pids = []
#         valid_pids = []
#         for pid in pids:
#             org_records = []
#             org_records_path = os.path.join(self.input_dir, pid, "RECORDS")
#             if not os.path.isfile(org_records_path): continue;
#             with open(org_records_path, 'r') as f:
#                 records = f.readlines()
#                 org_records.append([r.split("\n")[0] for r in records if r.find("_EEG")!=-1])
#             prcs_records = []
#             prcs_records_path = os.path.join(self.prev_dir, pid, "RECORDS")
#             if not os.path.isfile(prcs_records_path):
#                 invalid_pids.append(pid)
#                 continue
#             with open(prcs_records_path, 'r') as f:
#                 records = f.readlines()
#                 prcs_records.append([r for r in records])
#             if(len(prcs_records)!=len(org_records)):
#                 invalid_pids.append(pid)
#             else: 
#                 valid_pids.append(pid)
#         print(f'Invalid Patients: {len(invalid_pids)}')
#         print(f'Valid Patients: {len(valid_pids)}')

#     def split_dataset(self, labels_csv_path, output_dir, val_ratio=0.2):
#         """Split dataset into training and validation sets"""
#         df = pd.read_csv(labels_csv_path)
#         train_df, val_df = train_test_split(df, test_size=val_ratio, stratify=df['outcome'])
#         train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
#         val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
#         print(f"Training and validation sets saved to {output_dir}")

        

# if __name__ == "__main__":

#     preprocessor = EEGPreprocessor(
#         input_dir='/media/brainwave/2Tb HDD/physionet.org/files/i-care/2.1/training',
#         output_dir='/media/brainwave/2Tb HDD/preprocessed/',
#         prev_dir='/media/brainwave/Hard Disk/physionet.org/files/i-care/brainwave/v2/processed_training',
#         target_fs=100,
#         skip_processed=True,
#         records_per_patient=None
#     )
#     # preprocessor.process_dataset()
#     # preprocessor.generate_record_label_csv('labels.csv')
#     preprocessor.generate_output_csv('labels.csv')
#     preprocessor.split_dataset('labels.csv', 'processed_training', val_ratio=0.2)

#     # preprocessor.validate_dataset()

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
    label_mapping = {"Good": 0, "Poor": 1}

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