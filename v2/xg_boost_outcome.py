import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from xgboost import XGBClassifier

# Paths
FEATURES_JSON = "eeg_feature_dataset-full.json"
METADATA_JSON = "patient-metadata-full.json"
MODEL_OUTPUT = "xgboost_outcome.pkl"

# Custom weights for CPC labels
custom_weights = {
    0: 0.7,
    1: 1.0
}

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return {entry["Patient"]: entry for entry in metadata}

def process_metadata(metadata):
    age = metadata.get("Age", 0)
    sex = 1 if metadata.get("Sex") == "Male" else 0
    rosc = metadata.get("ROSC")
    rosc = 0 if rosc == "nan" or rosc is None else float(rosc)
    ohca = 1 if metadata.get("OHCA") else 0
    rhythm = 1 if metadata.get("Shockable Rhythm") else 0
    ttm = metadata.get("TTM", 0)

    return [age, sex, rosc, ohca, rhythm, ttm]

def build_feature_matrix():
    with open(FEATURES_JSON, "r") as f:
        feature_data = json.load(f)
    metadata_map = load_metadata(METADATA_JSON)

    eeg_keys = ["mean", "std", "var", "rms", "kurtosis", "power", "psd", "pfd", "pe"]
    metadata_keys = ["Age", "Sex", "ROSC", "OHCA", "Shockable Rhythm", "TTM"]
    all_feature_names = eeg_keys + metadata_keys

    feature_rows = []
    labels = []

    for record in feature_data:
        raw_id = record.get("patient_id")
        if raw_id is None:
            continue

        try:
            patient_id = int(raw_id.lstrip("0"))
        except ValueError:
            continue

        metadata_features = process_metadata(metadata_map.get(patient_id, {}))

        eeg_features = [record.get(key, 0) if record.get(key, 0) is not None else 0 for key in eeg_keys]
        eeg_features = [0 if isinstance(v, float) and np.isnan(v) else v for v in eeg_features]

        full_features = eeg_features + metadata_features
        feature_rows.append(full_features)
        labels.append(record["outcome"])

    X_df = pd.DataFrame(feature_rows, columns=all_feature_names)
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"Feature matrix shape: {X_df.shape}")
    y = np.array(labels)

    return X_df, y

def train_with_grid_search(X, y):
    # Define model with initial params including bagging
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42
    )

    # Parameter grid for GridSearch
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 10],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],              # Bagging: row sampling
        "colsample_bytree": [0.8, 1.0]        # Bagging: feature sampling
    }

    # Create weights
    sample_weights = [custom_weights[int(label)] for label in y]

    # K-Fold CV setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=skf,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X, y, sample_weight=sample_weights)

    print(f"\n✅ Best Parameters from Grid Search: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred = y_pred
    y_test = y_test

    print("\n✅ Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print(f"\n✅ Outcome Evaluation:")
    print(f"True Positive Rate (Sensitivity): {tpr:.3f}")
    print(f"False Positive Rate: {fpr:.3f}")

def main():
    X, y = build_feature_matrix()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = train_with_grid_search(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

    joblib.dump(clf, MODEL_OUTPUT)
    print(f"\n✅ Model saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()
