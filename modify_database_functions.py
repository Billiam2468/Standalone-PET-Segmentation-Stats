# Master python script to reorganize databases into form that works with segmentation/statistic scripts

import os
import shutil


# This function reorganizes a home directory of patient scans that have date subfolders inside rather than PET and CT folders directly inside the patient folder
# It will reorganize the database to remove the date subfolder and rename the patient folder instead with PET and CT folders directly inside
def date_subfolders_to_patient_date_name(home_dir):
    with os.scandir(home_dir) as patients:
        for patient in patients:
            if patient.is_dir():
                patient_name = patient.name
                patient_dir = os.path.join(home_dir, patient_name)
                
                with os.scandir(patient_dir) as dates:
                    for date in dates:
                        if date.is_dir():
                            date_name = date.name
                            date_dir = os.path.join(patient_dir, date_name)

                            # New directory in home_dir with format: patient_date
                            new_dir_name = f"{patient_name}_{date_name}"
                            new_dir_path = os.path.join(home_dir, new_dir_name)

                            # Create the new directory
                            os.makedirs(new_dir_path, exist_ok=True)

                            # Move all contents from date_dir to new_dir_path
                            for item in os.listdir(date_dir):
                                src = os.path.join(date_dir, item)
                                dst = os.path.join(new_dir_path, item)
                                shutil.move(src, dst)

                            # Optionally remove the now empty date_dir
                            os.rmdir(date_dir)

                # Optionally remove the now empty patient_dir
                # Check if only hidden files remain and remove them before deleting
                remaining_items = [f for f in os.listdir(patient_dir) if not f.startswith('.')]
                if not remaining_items:
                    # Optionally, remove hidden files too
                    for f in os.listdir(patient_dir):
                        try:
                            os.remove(os.path.join(patient_dir, f))
                        except IsADirectoryError:
                            shutil.rmtree(os.path.join(patient_dir, f))
                    os.rmdir(patient_dir)

#date_subfolders_to_patient_date_name("/Users/williamlee/Documents/Example/test copy")

# This function takes the current names of the CT and PET folders in a patient directory and renames it to the standard CT and PET
def rename_to_PET_and_CT(home_dir, ct_folder, pet_folder):
    with os.scandir(home_dir) as patients:
        for patient in patients:
            if patient.is_dir():
                patient_name = patient.name
                patient_dir = os.path.join(home_dir, patient_name)

                with os.scandir(patient_dir) as scans:
                    for scan in scans:
                        if scan.is_dir():
                            print(scan.name)
                            old_scan_path = os.path.join(patient_dir, scan.name)

                            if scan.name == ct_folder:
                                new_scan_path = os.path.join(patient_dir, "CT")
                                os.rename(old_scan_path, new_scan_path)
                                print(f"Renamed {scan.name} to CT in {patient_name}")

                            elif scan.name == pet_folder:
                                new_scan_path = os.path.join(patient_dir, "PET")
                                os.rename(old_scan_path, new_scan_path)
                                print(f"Renamed {scan.name} to PET in {patient_name}")

#rename_to_PET_and_CT("/Users/williamlee/Documents/Example/test copy 2", "old_CT", "old_PET")