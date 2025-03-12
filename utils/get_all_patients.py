import os
def get_all_patients(root_path):
    patient_ids = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    patient_ids = patient_ids[:500]
    patients_data = {}

    total_patient_count = 0

    for patient_id in patient_ids:
        records_file = os.path.join(root_path, patient_id, 'RECORDS')
        metadata_path = os.path.join(root_path, patient_id, f'{patient_id}.txt')

        if not os.path.isfile(records_file):
            continue

        try:
            with open(records_file, 'r') as f:
                records = [line.strip() for line in f if line.strip().split("_")[-1] == 'EEG']
            if(not os.path.exists(metadata_path)): 
                continue
            patients_data[patient_id] = records
            total_patient_count += 1

        except FileNotFoundError:
            continue
        
    
    print(f'Total Patients: {len(patients_data.keys())}')
    print(f'Total Records: {sum(len(records) for records in patients_data.values())}')

    print(f'Training Patients {patients_data.keys()}')

    return patients_data