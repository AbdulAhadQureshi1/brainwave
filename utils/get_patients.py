import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def get_patients(root_path, prototype=False, val_split=0.2, per_class=15, max_num_patients=None, plot_data_distribution=False):
    """Retrieves patient directories from the root path and splits data into training and validation sets."""
    patient_ids = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    patients_data = {}
    val_data = {}

    outcomes_count = [0, 0]
    good_patients = []
    poor_patients = []
    total_patient_count = 0
    val_outcomes_count = [0, 0]

    max_val_per_class = int(per_class * val_split)

    missing = []

    for patient_id in patient_ids:
        records_file = os.path.join(root_path, patient_id, 'RECORDS')
        metadata_path = os.path.join(root_path, patient_id, f'{patient_id}.txt')

        if not os.path.isfile(records_file):
            missing.append(patient_id)
            continue

        if (max_num_patients != None and total_patient_count >= max_num_patients):
            break

        try:
            with open(records_file, 'r') as f:
                records = [line.strip() for line in f if line.strip().split("_")[-1] == 'EEG']
            if(not os.path.exists(metadata_path)): 
                continue
            with open(metadata_path, 'r') as file:
                metadata = file.readlines()
                outcome = [line.split(':')[-1].strip() for line in metadata if 'Outcome' in line][0]
                if outcome == 'Good':
                    if(prototype and per_class > len(good_patients)):
                        outcomes_count[0] += 1
                        good_patients.append(patient_id)
                        patients_data[patient_id] = records
                    elif(prototype and max_val_per_class > val_outcomes_count[0]):
                        val_data[patient_id] = records
                        val_outcomes_count[0] += 1
                    elif not prototype:
                        outcomes_count[0] += 1
                        good_patients.append(patient_id)
                        patients_data[patient_id] = records
                        
                else:
                    if(prototype and per_class > len(poor_patients)):
                        outcomes_count[1] += 1
                        poor_patients.append(patient_id)
                        patients_data[patient_id] = records
                    elif(prototype and max_val_per_class > val_outcomes_count[1]):
                        val_data[patient_id] = records
                        val_outcomes_count[1] += 1
                    elif not prototype:
                        outcomes_count[1] += 1
                        poor_patients.append(patient_id)
                        patients_data[patient_id] = records
                total_patient_count += 1

        except FileNotFoundError:
            continue
        
    
    print(f'Total Patients: {len(patients_data.keys())}')
    print(f'Total Records: {sum(len(records) for records in patients_data.values())}')
    print(f'Outcomes [Good, Poor] {outcomes_count}')

    print(f'Training Patients {patients_data.keys()}')

    print(f'Validation Patients: {len(val_data.keys())}')
    print(f'Validation Outcomes [Good, Poor]: {val_outcomes_count}')

    print(f'Val Patients {val_data.keys()}')

    print(f'Missing: {missing}')

    for pid in missing:
        print(f'training/{pid}')

    if plot_data_distribution:
        # Plot class distribution in training and validation sets
        patient_outcomes = [1 if pid in good_patients else 0 for pid in patients_data]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(['Good', 'Poor'], [patient_outcomes.count(1), patient_outcomes.count(0)])
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.title('Training Data Distribution')
        
        plt.show()

    return patients_data, val_data



