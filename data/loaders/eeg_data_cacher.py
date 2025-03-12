import os
import numpy as np
import wfdb
from scipy import signal
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import h5py
import hashlib

class EEGDataCacher:
    def __init__(self, root_dir, fs=100, cache_dir='../processed'):
        self.root_dir = root_dir
        self.fs = fs
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_path(self):
        """Generate cache file path based on parameters"""
        params_str = f"{self.fs}"
        cache_hash = hashlib.md5(params_str.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"eeg_cache_{cache_hash}.h5")

    def cache_data(self, patients_data):
        """Cache all EEG data for faster loading during training"""
        cache_file = self.get_cache_path()
        
        if os.path.exists(cache_file):
            print(f"Cache already exists at {cache_file}")
            return cache_file
            
        print(f"Creating cache at {cache_file} (this may take a while)...")
        
        # Create data index
        index = self._create_index(patients_data)
        total_records = len(index)
        print(f"Found {total_records} valid records to process")
        
        # Process data in batches
        batch_size = 50  # Adjust based on available memory
        processed_count = 0
        successful_count = 0
        
        with h5py.File(cache_file, 'w') as f:
            for batch_start in range(0, total_records, batch_size):
                batch_end = min(batch_start + batch_size, total_records)
                batch_index = index[batch_start:batch_end]
                
                with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                    futures = []
                    for patient_id, record in batch_index:
                        futures.append(executor.submit(
                            self._process_single_record, 
                            patient_id, 
                            record
                        ))
                    
                    # Process results as they complete
                    for idx, future in enumerate(futures):
                        signal_data, label, cpc = future.result()
                        processed_count += 1
                        
                        if signal_data is not None:
                            record_idx = batch_start + idx
                            grp = f.create_group(str(record_idx))
                            grp.create_dataset('signal', data=signal_data)
                            grp.create_dataset('label', data=label)
                            grp.create_dataset('cpc', data=cpc)
                            grp.attrs['patient_id'] = batch_index[idx][0]
                            successful_count += 1
                        
                        # Print progress
                        progress = (processed_count / total_records) * 100
                        print(f"Progress: {processed_count}/{total_records} records processed ({progress:.1f}%) - Successfully cached: {successful_count}", end='\r')
                
                # Force garbage collection after each batch
                executor.shutdown(wait=True)
        
        print(f"\nCaching complete! Saved {successful_count}/{total_records} records to {cache_file}")
        if successful_count < total_records:
            print(f"Note: {total_records - successful_count} records were skipped due to processing errors")
        return cache_file

    def _create_index(self, patients_data):
        """Create an index of valid recordings"""
        index = []
        for patient_id, records in patients_data.items():
            for record in records:
                if not record.endswith("EEG"):
                    continue
                record = os.path.join(self.root_dir, patient_id, record)
                record_path = record + '.hea'
                if self._is_valid_recording(record_path):
                    index.append((patient_id, record))
        return index

    def _process_single_record(self, patient_id, record_path):
        """Process a single record and return preprocessed signal, label, and CPC"""
        try:
            eeg_signal, sampling_rate, _, label, cpc = self._read_eeg_data(record_path, patient_id)
            eeg_signal = self._preprocess_eeg_signal(eeg_signal, sampling_rate)
            return eeg_signal, label, cpc
        except Exception as e:
            print(f"Error processing record {record_path}: {e}")
            return None, None, None

    def _read_eeg_data(self, record_name, patient_id):
        """Read EEG data from file and extract metadata including CPC"""
        record = wfdb.rdrecord(record_name)
        sampling_rate = record.fs
        
        metadata_path = os.path.join(self.root_dir, patient_id, f"{patient_id}.txt")
        label_mapping = {"Good": 0, "Poor": 1}

        label = -1
        cpc = -1
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as file:
                metadata = file.readlines()
                # Extract outcome label
                outcome_lines = [line for line in metadata if "Outcome" in line]
                if outcome_lines:
                    outcome = outcome_lines[0].split(":")[-1].strip()
                    label = label_mapping.get(outcome, -1)
                
                # Extract CPC score
                cpc_lines = [line for line in metadata if "CPC" in line]
                if cpc_lines:
                    cpc_str = cpc_lines[0].split(":")[-1].strip()
                    try:
                        cpc = int(cpc_str)
                    except ValueError:
                        pass  # CPC remains -1 if conversion fails
        
        return record.p_signal, sampling_rate, record.p_signal.shape[1], label, cpc

    def _preprocess_eeg_signal(self, eeg_signal, sampling_rate):
        """Preprocess EEG signal"""
        # Resample
        if sampling_rate != self.fs:
            eeg_signal = signal.resample(eeg_signal, int(eeg_signal.shape[0] * (self.fs / sampling_rate)), axis=0)
        
        # Handle NaN values
        eeg_signal = np.nan_to_num(eeg_signal, nan=0.0)
        
        # Apply filtering
        nyq = 0.5 * self.fs
        b, a = signal.butter(5, [0.5/nyq, 40/nyq], btype="band")
        eeg_signal = signal.lfilter(b, a, eeg_signal, axis=0)
        
        # Normalize and standardize
        eeg_signal = (eeg_signal - np.mean(eeg_signal, axis=0)) / (np.std(eeg_signal, axis=0) + 1e-6)
        
        # Standardize channels
        if eeg_signal.shape[1] > 19:
            return eeg_signal[:, :19]
        elif eeg_signal.shape[1] < 19:
            padding = np.zeros((eeg_signal.shape[0], 19 - eeg_signal.shape[1]))
            return np.hstack((eeg_signal, padding))
        return eeg_signal

    def _is_valid_recording(self, header_path, min_duration=20):
        """Check if recording is valid based on duration"""
        try:
            with open(header_path, "r") as f:
                lines = f.readlines()
            first_line = lines[0].strip().split()
            num_samples = int(first_line[3])
            sampling_frequency = int(first_line[2])
            duration = num_samples / sampling_frequency
            return duration >= min_duration
        except Exception as e:
            print(f"Error reading header file {header_path}: {e}")
            return False