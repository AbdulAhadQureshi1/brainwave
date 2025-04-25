import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# Paths
FEATURES_JSON = "eeg_feature_dataset.json"
METADATA_JSON = "patient-metadata.json"
MODEL_OUTPUT = "xgboost_cpc.pkl"

# Custom weights for CPC labels
custom_weights = {
    0: 0.8,
    1: 1.5,
    2: 2.0,
    3: 1.8,
    4: 0.5
}

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    # Convert list of dicts into dict keyed by Patient ID
    return {entry["Patient"]: entry for entry in metadata}

def process_metadata(metadata):
    # Numeric conversions and categorical encoding
    age = metadata.get("Age", 0)
    sex = 1 if metadata.get("Sex") == "Male" else 0
    rosc = metadata.get("ROSC")
    rosc = 0 if rosc == "nan" or rosc is None else float(rosc)
    ohca = 1 if metadata.get("OHCA") else 0
    rhythm = 1 if metadata.get("Shockable Rhythm") else 0
    ttm = metadata.get("TTM", 0)

    return [age, sex, rosc, ohca, rhythm, ttm]

def main():
    # Load data
    with open(FEATURES_JSON, "r") as f:
        feature_data = json.load(f)
    metadata_map = load_metadata(METADATA_JSON)

    # Feature keys from EEG
    eeg_keys = ["mean", "std", "var", "rms", "kurtosis", "power", "psd", "pfd", "pe"]

    # New: column names
    metadata_keys = ["Age", "Sex", "ROSC", "OHCA", "Shockable Rhythm", "TTM"]
    all_feature_names = eeg_keys + metadata_keys

    feature_rows = []
    labels = []

    for record in feature_data:
        raw_id = record.get("patient_id")
        if raw_id is None:
            print("⚠️ Missing patient_id in EEG record.")
            continue

        try:
            patient_id = int(raw_id.lstrip("0"))  # Convert "0328" → 328
        except ValueError:
            print(f"⚠️ Invalid patient_id format: {raw_id}")
            continue

        if patient_id not in metadata_map:
            print(f"⚠️ Patient ID {patient_id} not found in metadata.")
            metadata_features = [0] * 6
        else:
            metadata_features = process_metadata(metadata_map[patient_id])

        eeg_features = []
        for key in eeg_keys:
            val = record.get(key, 0)
            if key == "kurtosis" and (val is None or np.isnan(val)):
                val = 0
            eeg_features.append(val)

        full_features = eeg_features + metadata_features
        feature_rows.append(full_features)
        labels.append(record["cpc"])

    # Convert to DataFrame
    X_df = pd.DataFrame(feature_rows, columns=all_feature_names)
    # Ensure all columns in the feature DataFrame are numeric
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    y = np.array(labels) - 1 # Shift because XG Boost needs 0-4

    print("\n🔍 Sample of features (X):")
    print(X_df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.4, random_state=42, stratify=y
    )

    # Generate sample weights from training labels
    sample_weights = [custom_weights[int(label)] for label in y_train]

    # Train model
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        objective="multi:softmax",  # Or "multi:softprob" if you want probabilities
        num_class=5,                # Replace with your number of CPC classes
        random_state=42,
        verbosity=1,
        eval_metric="mlogloss"
        # You can also tune `scale_pos_weight` or use sample weighting for imbalance
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred = y_pred + 1
    y_test = y_test + 1
    
    print("\n✅ Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Outcome
    # y_test_cpc and y_pred_cpc contain CPC values from 1 to 5
    y_true_outcome = np.where(y_test <= 2, 1, 0)   # Actual outcomes
    y_pred_outcome = np.where(y_pred <= 2, 1, 0)   # Predicted outcomes
    
    tn, fp, fn, tp = confusion_matrix(y_true_outcome, y_pred_outcome).ravel()

    tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn)  # False Positive Rate

    print(f"\n✅ Outcome Evaluation:")
    print(f"True Positive Rate (TPR / Sensitivity): {tpr:.3f}")
    print(f"False Positive Rate (FPR): {fpr:.3f}")

    # Save model
    joblib.dump(clf, MODEL_OUTPUT)
    print(f"\n✅ Model saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()
