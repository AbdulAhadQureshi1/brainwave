import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Load and extract the index_map list
with open("metadata.json", "r") as f:
    data = json.load(f)["index_map"]

# Group records by patient_id
patients = defaultdict(list)
for entry in data:
    patients[entry['patient_id']].append(entry)

# Shuffle patient_ids for randomness
all_patient_ids = list(patients.keys())
random.shuffle(all_patient_ids)

# Create stratification labels (by outcome and cpc)
stratify_labels = []
for pid in all_patient_ids:
    outcome = patients[pid][0]["outcome"]
    cpc = patients[pid][0]["cpc"]
    stratify_labels.append((outcome, cpc))

# Split into train (60%), temp (40%)
train_ids, temp_ids, _, temp_labels = train_test_split(
    all_patient_ids, stratify_labels, test_size=0.4, stratify=stratify_labels, random_state=42
)

# Split temp into val (20%) and test (20%)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.5, stratify=temp_labels, random_state=42
)

# Collect entries for each split
train_data = [entry for pid in train_ids for entry in patients[pid]]
val_data = [entry for pid in val_ids for entry in patients[pid]]
test_data = [entry for pid in test_ids for entry in patients[pid]]

# Save each to a file (as a list, not under index_map)
with open("train_metadata.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("validation_metadata.json", "w") as f:
    json.dump(val_data, f, indent=4)

with open("test_metadata.json", "w") as f:
    json.dump(test_data, f, indent=4)

print("✅ Metadata split complete!")
