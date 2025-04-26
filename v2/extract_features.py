import os
import json
import h5py
import numpy as np
from scipy.stats import kurtosis
from tqdm import tqdm
import concurrent.futures
from antropy import petrosian_fd, perm_entropy

# Paths
METADATA_PATH = "new-metadata.json"
DATA_DIR = "dataset/dataset"
OUTPUT_JSON = "lol-metadata.json"

def fast_extract_features(eeg_windows):
    """Feature extraction with real PFD and PE"""

    # Ensure batch dimension
    if len(eeg_windows.shape) == 2:
        eeg_windows = eeg_windows[np.newaxis, ...]
    
    means = np.mean(eeg_windows, axis=(1, 2))
    stds = np.std(eeg_windows, axis=(1, 2))
    vars_ = np.var(eeg_windows, axis=(1, 2))
    rms = np.sqrt(np.mean(eeg_windows ** 2, axis=(1, 2)))
    kurt = kurtosis(eeg_windows.reshape(eeg_windows.shape[0], -1), axis=1)
    power = np.mean(eeg_windows ** 2, axis=(1, 2))
    psd_approx = vars_

    pfd_vals = np.array([petrosian_fd(window.reshape(-1)) for window in eeg_windows])
    pe_vals = np.array([perm_entropy(window.reshape(-1), normalize=True) for window in eeg_windows])

    # No need for a loop if just 1 window
    feature_list = [{
        "mean": float(means[0]),
        "std": float(stds[0]),
        "var": float(vars_[0]),
        "rms": float(rms[0]),
        "kurtosis": float(kurt[0]),
        "power": float(power[0]),
        "psd": float(psd_approx[0]),
        "pfd": float(pfd_vals[0]),
        "pe": float(pe_vals[0])
    }]
    
    return feature_list

def process_record(record_info):
    patient_id = record_info["patient_id"]
    record_name = record_info["record"]
    record_path = os.path.join(DATA_DIR, patient_id, f"{record_name}.h5")

    try:
        with h5py.File(record_path, "r") as f:
            eeg_windows = f["windows"][:]  # shape: (n_windows, n_samples)

        all_window_features = fast_extract_features(eeg_windows)

        aggregated_features = {
            key: float(np.mean([f[key] for f in all_window_features]))
            for key in all_window_features[0]
        }

        result = {
            "patient_id": patient_id,
            "record": record_name,
            "outcome": record_info["outcome"],
            "cpc": record_info["cpc"],
            **aggregated_features
        }

    except Exception as e:
        print(f"❌ Error processing {record_name}: {e}")
        result = None

    return result

def main():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)["index_map"]

    features_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_record, metadata), total=len(metadata)))

    # Remove any failed results
    features_list = [res for res in results if res is not None]

    with open(OUTPUT_JSON, "w") as out_f:
        json.dump(features_list, out_f, indent=4)

    print(f"\n✅ All done! Features saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
