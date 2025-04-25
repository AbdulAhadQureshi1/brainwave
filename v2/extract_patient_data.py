import os
import json

# Root folder containing your dataset
data_dir = "D:/data"  # Change this to your actual path

# Fields to exclude
exclude_fields = {'Outcome', 'CPC'}

# Data storage
dataset = []

# Loop through folders and files
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            record = {}

            with open(file_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(":", 1)
                        key = key.strip()
                        value = value.strip()

                        if key not in exclude_fields:
                            # Try to convert to appropriate data type
                            if value.isdigit():
                                value = int(value)
                            elif value.lower() in ['true', 'false']:
                                value = value.lower() == 'true'
                            record[key] = value
            
            dataset.append(record)

# Save to JSON
output_path = "patient-metadata.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset saved to {output_path}")
