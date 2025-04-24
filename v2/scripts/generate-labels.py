import os
import json
import h5py

def create_metadata_file(processed_data_folder, original_data_folder, metadata_path):
    """
    Creates a metadata file by reading patient IDs from the processed data folder,
    extracting corresponding labels from the original data folder, and reading the number
    of windows from each processed record.

    Args:
        processed_data_folder (str): Path to the folder containing preprocessed HDF5 files.
        original_data_folder (str): Path to the folder containing original patient data.
        metadata_path (str): File path where the metadata JSON will be saved.
    """
    metadata = {'index_map': []}
    # Define outcome mapping as in the original preprocessing
    label_mapping = {"Good": 0, "Poor": 1}
    
    # Iterate through each patient folder in the processed data folder
    for patient_id in os.listdir(processed_data_folder):
        patient_processed_dir = os.path.join(processed_data_folder, patient_id)
        if not os.path.isdir(patient_processed_dir):
            continue
        
        # Locate the corresponding patient metadata file in the original data folder
        original_patient_dir = os.path.join(original_data_folder, patient_id)
        original_metadata_file = os.path.join(original_patient_dir, f"{patient_id}.txt")
        if not os.path.exists(original_metadata_file):
            print(f"Metadata file not found for patient {patient_id} in original data.")
            continue
        
        # Read labels from the original metadata file
        outcome, cpc = None, None
        try:
            with open(original_metadata_file, "r") as f:
                lines = f.readlines()
                # Look for lines containing "Outcome" and "CPC"
                for line in lines:
                    if "Outcome" in line:
                        outcome = line.split(":")[-1].strip()
                    elif "CPC" in line:
                        cpc = int(line.split(":")[-1].strip())
        except Exception as e:
            print(f"Error reading metadata for patient {patient_id}: {e}")
            continue
        
        if outcome is None or cpc is None:
            print(f"Missing outcome or CPC for patient {patient_id}.")
            continue
        
        # Process each preprocessed record for this patient
        for record_file in os.listdir(patient_processed_dir):
            # Process only HDF5 files
            if not record_file.endswith(".h5"):
                continue
            
            record_base = record_file[:-3]  # remove the ".h5" extension
            record_path = os.path.join(patient_processed_dir, record_file)
            
            # Open the HDF5 file and read the shape of the "windows" dataset
            try:
                with h5py.File(record_path, "r") as hf:
                    if "windows" not in hf:
                        print(f"'windows' dataset not found in {record_path}.")
                        continue
                    windows = hf["windows"]
                    num_windows = windows.shape[0]
            except Exception as e:
                print(f"Error reading {record_path}: {e}")
                continue
            
            # Add the record information to the metadata index_map
            metadata['index_map'].append({
                'patient_id': patient_id,
                'record': record_base,
                'num_windows': num_windows,
                'outcome': label_mapping.get(outcome, -1),
                'cpc': cpc
            })
    
    # Save metadata to a JSON file
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata file created at {metadata_path}.")
    except Exception as e:
        print(f"Error writing metadata file: {e}")

# Example usage:
create_metadata_file(
    processed_data_folder='../preprocessed_data',
    original_data_folder='/media/brainwave/2Tb HDD/physionet.org/files/i-care/2.1/training',
    metadata_path='../metadata.json'
)
