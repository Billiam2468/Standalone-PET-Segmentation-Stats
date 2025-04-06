# Master python script to reorganize databases into form that works with segmentation/statistic scripts

import os
import shutil

# def date_subfolders_to_patient_date_name(home_dir):
#     with os.scandir(home_dir) as patients:
#         for patient in patients:
#             if patient.is_dir():
#                 patient_dir = os.path.join(home_dir, patient)
#                 with os.scandir(patient_dir) as dates:
#                     for date in dates:
#                         if date.is_dir():
#                             print(date.name)


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

date_subfolders_to_patient_date_name("/Users/williamlee/Documents/Example/test copy")