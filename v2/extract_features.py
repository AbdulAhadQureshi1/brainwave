import os
import json
import h5py
import numpy as np
from scipy.stats import kurtosis
from tqdm import tqdm
import concurrent.futures
from antropy import petrosian_fd, perm_entropy

# Paths
METADATA_PATH = "dataset.json"
DATA_DIR = "preprocessed_data"
OUTPUT_JSON = "eeg_feature_dataset.json"

# def fast_extract_features(eeg_windows):
#     """Fast feature extraction without slow Welch, PFD, PE"""
#     means = np.mean(eeg_windows, axis=1)
#     stds = np.std(eeg_windows, axis=1)
#     vars_ = np.var(eeg_windows, axis=1)
#     rms = np.sqrt(np.mean(eeg_windows ** 2, axis=1))
#     kurt = kurtosis(eeg_windows, axis=1)
#     power = np.mean(eeg_windows ** 2, axis=1)
#     psd_approx = vars_  # Approximate PSD by variance

#     feature_list = []
#     for i in range(eeg_windows.shape[0]):
#         feature_list.append({
#             "mean": float(means[i]),
#             "std": float(stds[i]),
#             "var": float(vars_[i]),
#             "rms": float(rms[i]),
#             "kurtosis": float(kurt[i]),
#             "power": float(power[i]),
#             "psd": float(psd_approx[i]),
#             # You can remove pfd and pe here to speed up more
#             "pfd": 0.0,
#             "pe": 0.0
#         })

#     return feature_list

def fast_extract_features(eeg_windows):
    """Feature extraction with real PFD and PE"""
    means = np.mean(eeg_windows, axis=1)
    stds = np.std(eeg_windows, axis=1)
    vars_ = np.var(eeg_windows, axis=1)
    rms = np.sqrt(np.mean(eeg_windows ** 2, axis=1))
    kurt = kurtosis(eeg_windows, axis=1)
    power = np.mean(eeg_windows ** 2, axis=1)
    psd_approx = vars_

    # Real PFD and PE
    pfd_vals = np.array([petrosian_fd(window) for window in eeg_windows])
    pe_vals = np.array([perm_entropy(window, normalize=True) for window in eeg_windows])

    feature_list = []
    for i in range(eeg_windows.shape[0]):
        feature_list.append({
            "mean": float(means[i]),
            "std": float(stds[i]),
            "var": float(vars_[i]),
            "rms": float(rms[i]),
            "kurtosis": float(kurt[i]),
            "power": float(power[i]),
            "psd": float(psd_approx[i]),
            "pfd": float(pfd_vals[i]),
            "pe": float(pe_vals[i])
        })

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
