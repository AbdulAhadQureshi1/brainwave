import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
FEATURES_JSON = "eeg_feature_dataset.json"
MODEL_OUTPUT = "random_forest_cpc_model.pkl"

# Define custom class weights
custom_weights = {
    1: 2.0,
    2: 2.0,
    3: 1.5,
    4: 1.0,
    5: 0.5   # 💥 Give CPC 5 less weight
}

def main():
    # Load extracted features
    with open(FEATURES_JSON, "r") as f:
        feature_data = json.load(f)

    # Select features
    feature_keys = ["mean", "std", "var", "rms", "kurtosis", "power", "psd", "pfd", "pe"]

    # Build X (features) and y (labels = CPC scores)
    X = np.array([[record[key] for key in feature_keys] for record in feature_data])
    y = np.array([record["cpc"] for record in feature_data])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Train RandomForest
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight=custom_weights)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluation
    print("\n✅ Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(clf, MODEL_OUTPUT)
    print(f"\n✅ Model saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()
