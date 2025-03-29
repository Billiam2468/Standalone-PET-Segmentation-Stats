import nibabel as nib
import dicom2nifti as d2n
import os
import pydicom
import PySimpleGUI as sg
import shutil

import numpy as np
from datetime import datetime
import time
import nrrd
import gc
import sys
import csv
from scipy.ndimage import zoom
from scipy import stats

import SimpleITK as sitk




total = [
    "background",
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "esophagus",
    "trachea",
    "thyroid_gland",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "kidney_cyst_left",
    "kidney_cyst_right",
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "atrial_appendage_left",
    "superior_vena_cava",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "spinal_cord",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "autochthon_left",
    "autochthon_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "brain",
    "skull",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "sternum",
    "costal_cartilages"
]

lung_vessels = [
    "background",
    "lung_vessels",
    "lung_trachea_bronchia"
]

body = [
    "background",
    "body_trunc",
    "body_extremities"
]

cerebral_bleed = [
    "background",
    "intracerebral_hemorrhage"
]

hip_implant = [
    "background",
    "hip_implant"
]

coronary_arteries = [
    "background",
    "coronary_arteries"
]

pleural_pericard_effusion = [
    "background",
    "pleural_effusion",
    "pericardial_effusion"
]

head_glands_cavities = [
    "background",
    "eye_left",
    "eye_right",
    "eye_lens_left",
    "eye_lens_right",
    "optic_nerve_left",
    "optic_nerve_right",
    "parotid_gland_left",
    "parotid_gland_right",
    "submandibular_gland_right",
    "submandibular_gland_left",
    "nasopharynx",
    "oropharynx",
    "hypopharynx",
    "nasal_cavity_right",
    "nasal_cavity_left",
    "auditory_canal_right",
    "auditory_canal_left",
    "soft_palate",
    "hard_palate"
]

head_muscles = [
    "background",
    "masseter_right",
    "masseter_left",
    "temporalis_right",
    "temporalis_left",
    "lateral_pterygoid_right",
    "lateral_pterygoid_left",
    "medial_pterygoid_right",
    "medial_pterygoid_left",
    "tongue",
    "digastric_right",
    "digastric_left"
]

headneck_bones_vessels = [
    "background",
    "larynx_air",
    "thyroid_cartilage",
    "hyoid",
    "cricoid_cartilage",
    "zygomatic_arch_right",
    "zygomatic_arch_left",
    "styloid_process_right",
    "styloid_process_left",
    "internal_carotid_artery_right",
    "internal_carotid_artery_left",
    "internal_jugular_vein_right",
    "internal_jugular_vein_left"
]

headneck_muscles = [
    "background",
    "sternocleidomastoid_right",
    "sternocleidomastoid_left",
    "superior_pharyngeal_constrictor",
    "middle_pharyngeal_constrictor",
    "inferior_pharyngeal_constrictor",
    "trapezius_right",
    "trapezius_left",
    "platysma_right",
    "platysma_left",
    "levator_scapulae_right",
    "levator_scapulae_left",
    "anterior_scalene_right",
    "anterior_scalene_left",
    "middle_scalene_right",
    "middle_scalene_left",
    "posterior_scalene_right",
    "posterior_scalene_left",
    "sterno_thyroid_right",
    "sterno_thyroid_left",
    "thyrohyoid_right",
    "thyrohyoid_left",
    "prevertebral_right",
    "prevertebral_left"
]

liver_vessels = [
    "background",
    "liver_vessels",
    "liver_tumor"
]

oculomotor_muscles = [
    "background",
    "skull",
    "eyeball_right",
    "lateral_rectus_muscle_right",
    "superior_oblique_muscle_right",
    "levator_palpebrae_superioris_right",
    "superior_rectus_muscle_right",
    "medial_rectus_muscle_left",
    "inferior_oblique_muscle_right",
    "inferior_rectus_muscle_right",
    "optic_nerve_left",
    "eyeball_left",
    "lateral_rectus_muscle_left",
    "superior_oblique_muscle_left",
    "levator_palpebrae_superioris_left",
    "superior_rectus_muscle_left",
    "medial_rectus_muscle_right",
    "inferior_oblique_muscle_left",
    "inferior_rectus_muscle_left",
    "optic_nerve_right"
]

heartchambers_highres = [
    "background",
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
    "aorta",
    "pulmonary_artery"
]

appendicular_bones = [
    "background",
    "patella",
    "tibia",
    "fibula",
    "tarsal",
    "metatarsal",
    "phalanges_feet",
    "ulna",
    "radius",
    "carpal",
    "metacarpal",
    "phalanges_hand"
]

tissue_types = [
    "background",
    "subcutaneous_fat",
    "torso_fat",
    "skeletal_muscle"
]

brain_structures = [
    "background",
    "brainstem",  # + brain_parenchyma
    "subarachnoid_space",
    "venous_sinuses", # + dural folds
    "septum_pellucidum",
    "cerebellum",
    "caudate_nucleus",
    "lentiform_nucleus",
    "insular_cortex",
    "internal_capsule",
    "ventricle",
    "central_sulcus",
    "frontal_lobe",
    "parietal_lobe",
    "occipital_lobe",
    "temporal_lobe",
    "thalamus"
]

vertebrae_body = [
    "background",
    "vertebrae_body"
]

face = [
    "background",
    "face"
]


def save_to_csv(data, filename):
    # Initialize a list to hold the header and rows
    header = ['Patient']
    rows = []

    # Extract organ statistics
    for patient, organs in data.items():
        row = [patient]
        for organ, stats in organs.items():
            # Create keys for each statistic
            for stat_name, value in stats.items():
                key = f"{organ}_{stat_name}"
                row.append(value)  # Append the statistic value
                # Add the key to the header if it's not already there
                if key not in header:
                    header.append(key)
        
        rows.append(row)

    # Write to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        for row in rows:
            # Ensure the row has the same number of columns as the header
            # Fill missing values with None
            row_dict = {header[i]: row[i] for i in range(len(row))}
            full_row = [row_dict.get(col, None) for col in header]
            writer.writerow(full_row)

def old_upscale_suv_values_3d(suv_values, new_shape):
    # Calculate the zoom factors for each dimension
    zoom_factors = [n / o for n, o in zip(new_shape, suv_values.shape)]
    
    # Perform the upscaling using ndimage.zoom
    suv_interpolated = zoom(suv_values, zoom_factors, order=1)  # 'order=1' for linear interpolation
    
    return suv_interpolated.astype(np.float32)

def upscale_pet_to_segmentation_with_voxel_spacing(pet_suv_vals, pet_shape, segmentation_shape, pet_spacing, segmentation_spacing):
    """
    Upscale PET suv values to match segmentation shape, accounting for both voxel spacing and shape differences.
    """
    # Calculate zoom factors for each dimension (x, y, z)
    zoom_factors = [
        (segmentation_shape[0] / pet_shape[0]) * (pet_spacing[0] / segmentation_spacing[0]),  # x dimension
        (segmentation_shape[1] / pet_shape[1]) * (pet_spacing[1] / segmentation_spacing[1]),  # y dimension
        (segmentation_shape[2] / pet_shape[2]) * (pet_spacing[2] / segmentation_spacing[2])   # z dimension
    ]
    
    print(zoom_factors)

    # Perform the upscaling using ndimage.zoom (order=1 for linear interpolation)
    upscaled_pet = zoom(pet_suv_vals, zoom_factors, order=1)  # Linear interpolation
    
    return upscaled_pet.astype(np.float32)

    print("shape of interp", upscaled_pet.shape)
    
    return upscaled_pet.astype(np.float32)

def olde_upscale_suv_values_3d(suv_values, original_spacing, new_shape, new_spacing, interpolation_order=1):
    print("values")
    print(original_spacing)
    print(new_shape)
    print(new_spacing)
    """
    Upscale 3D SUV values (e.g., from a PET scan) to a new shape while accounting for voxel spacing.
    
    Parameters:
    - suv_values (np.ndarray): Original 3D SUV array.
    - original_spacing (tuple): Voxel spacing of the original array (e.g., (x_spacing, y_spacing, z_spacing)).
    - new_shape (tuple): Desired shape of the upscaled array (e.g., (new_x, new_y, new_z)).
    - new_spacing (tuple): Voxel spacing of the new array (e.g., (new_x_spacing, new_y_spacing, new_z_spacing)).
    - interpolation_order (int): Interpolation order for upscaling (0 for nearest-neighbor, 1 for linear, etc.)
    
    Returns:
    - np.ndarray: Upscaled SUV values.
    """
    # Calculate zoom factors based on the shape change
    zoom_factors_shape = [n / o for n, o in zip(new_shape, suv_values.shape)]
    
    # Calculate zoom factors based on voxel spacing change
    zoom_factors_spacing = [o / n for n, o in zip(original_spacing, new_spacing)]
    
    # Zoom factors for each axis should be calculated separately
    zoom_factors = [f_shape * f_spacing for f_shape, f_spacing in zip(zoom_factors_shape, zoom_factors_spacing)]
    
    # Perform the upscaling using ndimage.zoom with the selected interpolation method
    suv_interpolated = zoom(suv_values, zoom_factors, order=interpolation_order)  # Adjust interpolation method

    print("shape of interpolated", suv_interpolated.shape)
    
    return suv_interpolated.astype(np.float32)


import dicom_numpy

def get_dicom_files_from_folder(folder_path):
    """
    Returns a list of paths to DICOM files in the given folder.
    """
    dicom_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            try:
                # Verify if the file is a valid DICOM file
                pydicom.dcmread(file_path, stop_before_pixels=True)
                dicom_files.append(file_path)
            except pydicom.errors.InvalidDicomError:
                # Skip non-DICOM files
                continue
    return dicom_files

def extract_voxel_data(list_of_dicom_files):
    datasets = [pydicom.read_file(f) for f in list_of_dicom_files]
    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
    except dicom_numpy.DicomImportException as e:
        # invalid DICOM data
        raise
    return voxel_ndarray

def find_indices_in_array(A, B):
    """
    Given two arrays A and B, returns the 3D indices in B where the values are found in A.

    Parameters:
    - A (numpy.ndarray): Array containing values to search for.
    - B (numpy.ndarray): 3D array in which to search for values of A.

    Returns:
    - list of tuples: Each tuple contains the (x, y, z) coordinates where the value in B is in A.
    """
    # Use np.isin to find where elements of B are in A
    indices = np.where(np.isin(B, A))  # Returns a tuple of arrays with indices
    return list(zip(*indices))  # Combine the indices into (x, y, z) tuples

def more_optimized_suv_statistics(roi_data, pet_data, reference):
    # Flatten arrays for faster processing
    flat_roi_data = roi_data.ravel()
    flat_pet_data = pet_data.ravel()

    # Filter out background
    mask = flat_roi_data > 0
    flat_roi_data = flat_roi_data[mask]
    flat_pet_data = flat_pet_data[mask]

    # Unique ROI values
    unique_rois = np.unique(flat_roi_data)
    
    suv_stats = {}
    for roi in unique_rois:
        suv_values = flat_pet_data[flat_roi_data == roi]
        # if(roi == 57):
        #     indices = np.where(roi_data == 57)  # Returns a tuple of arrays with indices
        #     print(list(zip(*indices)))  # Combine the indices into (x, y, z) tuples
        #     print(np.where(roi_data == 57))
        #     print(suv_values.size)
        #     print(pet_data[58, 61, 355])
        #     indices = find_indices_in_array(suv_values, pet_data)
        #     print(indices)
        if suv_values.size > 0:
            suv_stats[reference[int(roi)]] = {
                'mean': np.mean(suv_values),
                'max': np.max(suv_values),
                'median': np.median(suv_values),
                'num_val': suv_values.size,
            }

    return suv_stats

def rename_PET_nifti(pet_nifti_path):
    parent_folder = os.path.basename(os.path.dirname(pet_nifti_path))

    parent_folder_path = os.path.dirname(os.path.dirname(pet_nifti_path))

    new_path = os.path.join(parent_folder_path, parent_folder + ".nii.gz")
    print(new_path)
    shutil.move(pet_nifti_path, new_path)
    os.rmdir(os.path.dirname(pet_nifti_path))

# Takes the relevant metadata from the DICOM file including the directory it came from and moves it into the metadata of the NIFTI file
def move_DICOM_Metadata_To_NIFTI(dicom_path, nifti_path, dicom_folder):
    ds = pydicom.dcmread(dicom_path)
    seriesTime = ds.SeriesTime
    injectionTime = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    halfLife = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    injectedDose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    patientWeight = ds.PatientWeight
    numSlices = ds.NumberOfSlices

    nifti_file = nib.load(nifti_path)

    nifti_file.header['descrip'] = f"{seriesTime}x{injectionTime}x{halfLife}x{injectedDose}x{patientWeight}"
    directory_binary = dicom_folder.encode('utf-8')  # Convert to bytes
    nifti_file.header.extensions.append(nib.nifti1.Nifti1Extension(code=6, content=directory_binary))

    nib.save(nifti_file, nifti_path)

def dicom2nifti(home_dir, output_dir):
    with os.scandir(home_dir) as patients:
        for patient in patients:
            if patient.is_dir():
                scan_dir = os.path.join(home_dir, patient, "PET")
                if os.path.exists(scan_dir):
                    save_dir = os.path.join(output_dir, patient.name)
                    os.makedirs(save_dir, exist_ok=True)
                    d2n.convert_directory(scan_dir, save_dir)

                    dicom_file = None
                    for file in os.listdir(scan_dir):
                        if file.startswith("."):
                            continue
                        else:
                            dicom_file = file
                            break
                    dicom_path = os.path.join(scan_dir, dicom_file)

                    nifti_file = None
                    for file in os.listdir(save_dir):
                        if file.startswith("."):
                            continue
                        else:
                            nifti_file = file
                            break
                    nifti_path = os.path.join(save_dir, nifti_file)

                    # print(dicom_path)
                    # print(nifti_path)

                    move_DICOM_Metadata_To_NIFTI(dicom_path, nifti_path, scan_dir)
                    rename_PET_nifti(nifti_path)
                else:
                    print(f"PET folder doesn't exist in {patient.name}")

def load_nifti_file(filepath):
    """Load a NIfTI file and return the data array."""
    nifti_img = nib.load(filepath)
    data = nifti_img.get_fdata()
    return data

def calculate_time_difference(scan_time_str, injection_time_str):
    # Define the correct time format
    time_format_with_microseconds = "%H%M%S.%f"
    time_format_without_microseconds = "%H%M%S"

    # Parse the time strings with and without microseconds
    try:
        scan_time = datetime.strptime(scan_time_str, time_format_with_microseconds)
    except ValueError:
        scan_time = datetime.strptime(scan_time_str, time_format_without_microseconds)

    try:
        injection_time = datetime.strptime(injection_time_str, time_format_with_microseconds)
    except ValueError:
        injection_time = datetime.strptime(injection_time_str, time_format_without_microseconds)

    # Remove the fractional seconds by setting microseconds to zero
    scan_time = scan_time.replace(microsecond=0)
    
    # Subtract the two datetime objects
    time_difference = scan_time - injection_time

    # Get the total difference in seconds
    total_seconds = time_difference.total_seconds()

    # We are using 40 minutes for dynamic scans. Will see how it affects values
    #return 2400
    return total_seconds

# Extracts the stored file directory in the NIFTI file header
def retrieve_file_directory(pet_nifti):
    nifti_file = nib.load(pet_nifti)
    header = nifti_file.header
     # Look for the directory path in custom extensions
    for ext in header.extensions:
        content = ext.get_content().decode('utf-8')  # Convert bytes back to string
        if content.startswith("/"):  # Assuming directory paths start with '/'
            return content


def compare_arrays(arr1, arr2, tolerance=1e-2):
    # Ensure arrays are of the same size
    if arr1.shape != arr2.shape:
        print("Arrays are of different sizes.")
        return

    # Compare element-wise with tolerance
    comparison = np.isclose(arr1, arr2, atol=tolerance)

    # Calculate the percentage of identical values
    identical_count = np.sum(comparison)
    total_elements = arr1.size
    identical_percentage = (identical_count / total_elements) * 100

    print(f"The arrays are {identical_percentage:.2f}% identical.")

    # Show indexes and values where arrays don't match
    mismatches = np.where(comparison == False)
    if len(mismatches[0]) > 0:
        print("Indexes where the arrays do not match (with tolerance):")
        for idx in zip(*mismatches):  # Unpack the 3D indices
            print(f"Index: {idx}, arr1 value: {arr1[idx]}, arr2 value: {arr2[idx]}")
    else:
        print("No mismatches found.")

def convert_raw_PET_to_SUV(pet_nifti):
    PET_data = None #load_nifti_file(pet_nifti)

    nifti_file = nib.load(pet_nifti)
    description = nifti_file.header['descrip'].tobytes().decode('ascii').split("x")
    description = [value.rstrip('\x00') for value in description]
    print(description)

    directory = retrieve_file_directory(pet_nifti)
    print(directory)

    dicom_files = get_dicom_files_from_folder(directory)
    if dicom_files:
        PET_data = extract_voxel_data(dicom_files)
        print("PET data shape:", PET_data.shape)
    else:
        print("No valid DICOM files found in the folder.")


    #aux_file = nifti_file.header['aux_file'].tobytes().decode('ascii')
    #print(aux_file)

    series_time = description[0]
    injection_time = description[1]
    half_life = float(description[2])
    injected_dose = float(description[3])
    patient_weight = float(description[4])*1000

    print("Series time ", series_time)
    print("injection time ", injection_time)
    print("half_life ", half_life)
    print("injected dose ", injected_dose)
    print("patient weight ", patient_weight)

    time_diff = abs(calculate_time_difference(series_time, injection_time))

    # This factor accounts for decay of injected dose
    decay_correction_factor = np.exp(-np.log(2) * ((time_diff/half_life)))

    # SUV = img data/(injected dose/body weight) = (img data * bodyweight)/(injected dose)
    SUV_factor = (patient_weight) / (injected_dose * decay_correction_factor)

    print("SUV_factor ", SUV_factor)
    final = (PET_data * SUV_factor).astype(np.float32)


    realData = load_nifti_file("/Users/williamlee/Desktop/realSUV.nii.gz")

    compare_arrays(final, realData)

    print(final[80, 80, 100])

    return final

# Function that takes the path of NIFTIs, creates a reference to its DICOM for metadata, and returns a dictionary of SUV values for each PET NIFTI
def extractSUVs(nifti_path):
    SUV_vals = {}
    metadata = {}

    with os.scandir(nifti_path) as entries:
        for entry in entries:
            nifti_dir = os.path.join(nifti_path, entry)
            if entry.is_file():
                # Ignore hidden files:
                if entry.name[0] == ".":
                    continue

                fileName = entry.name.removesuffix(".nii.gz")

                SUV_vals[fileName] = convert_raw_PET_to_SUV(nifti_dir)


                #nifti_img = nib.load(nifti_dir)
                nifti_img = sitk.ReadImage(nifti_dir)

                print("this spacings", nifti_img.GetSpacing())
                metadata[fileName] = {"spacing": nifti_img.GetSpacing(), "origin": nifti_img.GetOrigin(), "direction": nifti_img.GetDirection()}
    return SUV_vals, metadata

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# def visualize_slice(arr, slice_idx):
#     """Visualizes a specific slice of the segmentation array."""
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     # Show the 2D slice of the 3D array
#     ax.imshow(arr[slice_idx, :, :], cmap='tab20', interpolation='nearest')
    
#     # Add title and labels
#     ax.set_title(f'Slice {slice_idx}')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
    
#     plt.show()

# def scroll_through_3d_array(arr):
#     """Interactive slider to scroll through the slices of a 3D segmentation array."""
    
#     # Check if the array is 3D
#     if arr.ndim != 3:
#         raise ValueError("Input array must be 3D")

#     # Initialize the index of the slice to show
#     slice_idx = 0
    
#     # Create the initial plot to display a slice
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     # Function to update the displayed slice
#     def update(val):
#         nonlocal slice_idx
#         slice_idx = int(val)
#         ax.clear()  # Clear the axis to update the slice
#         ax.imshow(arr[slice_idx, :, :], cmap='tab20', interpolation='nearest')
#         ax.set_title(f'Slice {slice_idx}')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         fig.canvas.draw()

#     # Create a slider to scroll through slices
#     ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
#     slider = Slider(ax_slider, 'Slice', 0, arr.shape[0]-1, valinit=0, valstep=1)
#     slider.on_changed(update)
    
#     # Show the plot with the slider
#     plt.show()



def visualize_slice(arr, overlay_arr, slice_idx):
    """Visualizes a specific slice of the segmentation array with an overlay."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Show the 2D slice of the segmentation array
    ax.imshow(arr[slice_idx, :, :], cmap='tab20', interpolation='nearest', alpha=0.6)
    
    # Overlay the second array on top with transparency (alpha)
    ax.imshow(overlay_arr[slice_idx, :, :], cmap='hot', interpolation='nearest', alpha=0.4)
    
    # Add title and labels
    ax.set_title(f'Slice {slice_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.show()

def scroll_through_3d_array(arr, overlay_arr):
    """Interactive slider to scroll through the slices of a 3D segmentation array with an overlay."""
    
    # Check if both arrays are 3D and have the same shape
    if arr.ndim != 3 or overlay_arr.ndim != 3:
        raise ValueError("Both input arrays must be 3D")
    if arr.shape != overlay_arr.shape:
        raise ValueError("Both arrays must have the same shape")
    
    # Initialize the index of the slice to show
    slice_idx = 0
    
    # Create the initial plot to display a slice
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Function to update the displayed slice
    def update(val):
        nonlocal slice_idx
        slice_idx = int(val)
        ax.clear()  # Clear the axis to update the slice
        ax.imshow(arr[slice_idx, :, :], cmap='tab20', interpolation='nearest', alpha=0.6)
        ax.imshow(overlay_arr[slice_idx, :, :], cmap='hot', interpolation='nearest', alpha=0.4)
        ax.set_title(f'Slice {slice_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.canvas.draw()

    # Create a slider to scroll through slices
    ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Slice', 0, arr.shape[0]-1, valinit=0, valstep=1)
    slider.on_changed(update)
    
    # Show the plot with the slider
    plt.show()

def statistics_from_rois(segmentation_dir, pet_dir, extension, name_reference):

    stats = {}

    # #to_do = ['joints_10_FDG180.nrrd', 'joints_11_FDG180.nrrd', 'joints_37_FDG180.nrrd']

    counter = 1

    with os.scandir(segmentation_dir) as segmentations:
        for segmentation in segmentations:
            if segmentation.is_file():
                # Ignore hidden files:
                if segmentation.name[0] == ".":
                    continue
                
                num_segs = len(os.listdir(segmentation_dir))

                print(f"Working on segmentation {counter} of {num_segs}")
                print(segmentation.name)

                # Remove extension
                pet_name = segmentation.name.removesuffix(extension)
                
                seg_dir = os.path.join(segmentation_dir, segmentation.name)
                pet_dir = os.path.join(pet_dir, segmentation.name)

                print("seg dir", seg_dir)
                print("petdir", pet_dir)


                segmentation = sitk.ReadImage(seg_dir)
                pet = sitk.ReadImage(pet_dir)

                

                # Resample segmentation to PET scan resolution
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(pet)  # Use PET as reference for spacing and size
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for labels
                resampled_segmentation = resampler.Execute(segmentation)

                # Save the resampled segmentation
                #sitk.WriteImage(resampled_segmentation, 'resampled_segmentation.nii.gz')




                # #If files are nifti use below:
                # segmentation_img = nib.load(seg_dir)
                # segmentation_data = segmentation_img.get_fdata()
                # segmentation_voxel_spacing = segmentation_img.header.get_zooms()
                # print("seg voxel spacings, ", segmentation_voxel_spacing)



                # # Load segmentation file
                # segmentation = sitk.ReadImage(seg_dir)

                # # Convert the numpy array to a SimpleITK image
                # pet_image = sitk.GetImageFromArray(SUV_vals[pet_name])

                # # Set metadata for the PET image (spacing, origin, direction)
                # # You might need to replace these with the actual values
                # pet_image.SetSpacing(metadata[pet_name]["spacing"])       # Use the same spacing as the segmentation
                # pet_image.SetOrigin(metadata[pet_name]["origin"])         # Align origin with segmentation
                # pet_image.SetDirection(metadata[pet_name]["direction"])   # Align direction with segmentation

                # # Create a resampler
                # resampler = sitk.ResampleImageFilter()
                # resampler.SetReferenceImage(segmentation)  # Use the segmentation image as the reference
                # resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for segmentation labels
                # resampled_pet = resampler.Execute(pet_image)



                # # Save the resampled segmentation
                # sitk.WriteImage(resampled_pet, 'resampled_segmentation.nii.gz')

                # Convert the resampled segmentation back to a numpy array
                resampled_segmentation_array = sitk.GetArrayFromImage(resampled_segmentation)

                pet_array = sitk.GetArrayFromImage(pet)


                # resampled_segmentation_array = np.transpose(resampled_segmentation_array, (2, 1, 0))  # Example swap




                print("shape of resampled", resampled_segmentation_array.shape)
                print("shape of pet", pet_array.shape)


                # # Mirror SUV vals
                # SUV_vals[pet_name] = SUV_vals[pet_name][:, ::-1, :]

                

                # # Upscale the PET SUV values to match the shape of the segmentation
                # new_shape = (segmentation_data.shape[0], segmentation_data.shape[1], segmentation_data.shape[2])
                # print("Shape of segmentation is: ", new_shape)
                # print("pet_name is:", pet_name)
                # print("Shape of PET suv_vals is: ", SUV_vals[pet_name].shape)

                # # suv_values, segmentation_shape, pet_shape, pet_spacing, segmentation_spacing
                # upscaled_suv_values = old_upscale_suv_values_3d(SUV_vals[pet_name], new_shape)
                # #upscaled_suv_values = upscale_suv_values_3d(SUV_vals[pet_name], new_shape, SUV_vals[pet_name].shape, spacings[pet_name], segmentation_voxel_spacing) #spacings[pet_name], new_shape, )#upscale_suv_values_3d(SUV_vals[pet_name], new_shape)
                # #upscaled_suv_values = upscale_pet_to_segmentation_with_voxel_spacing(SUV_vals[pet_name], SUV_vals[pet_name].shape, new_shape, spacings[pet_name], segmentation_voxel_spacing)
                # print("finished upscaling")

                # downscaled_seg = old_upscale_suv_values_3d(segmentation_data, SUV_vals[pet_name].shape)

                #scroll_through_3d_array(resampled_segmentation_array, pet_array)
                #scroll_through_3d_array(res, pet_array)
                stats[pet_name] = more_optimized_suv_statistics(roi_data=resampled_segmentation_array, pet_data=pet_array, reference=name_reference)

                del resampled_segmentation_array, pet_array
                gc.collect()
                counter = counter + 1
    return stats

def old_statistics_from_rois(segmentation_dir, SUV_vals, spacings, extension, rois, name_reference):
    stats = {}

    # #to_do = ['joints_10_FDG180.nrrd', 'joints_11_FDG180.nrrd', 'joints_37_FDG180.nrrd']

    counter = 1

    with os.scandir(segmentation_dir) as segmentations:
        for segmentation in segmentations:
            if segmentation.is_file():
                # Ignore hidden files:
                if segmentation.name[0] == ".":
                    continue
                
                num_segs = len(os.listdir(segmentation_dir))

                print(f"Working on segmentation {counter} of {num_segs}")
                print(segmentation.name)

                # Remove extension
                pet_name = segmentation.name.removesuffix(extension)
                seg_dir = os.path.join(segmentation_dir, segmentation)
                

                #If files are nifti use below:
                segmentation_img = nib.load(seg_dir)
                segmentation_data = segmentation_img.get_fdata()
                segmentation_voxel_spacing = segmentation_img.header.get_zooms()
                print("seg voxel spacings, ", segmentation_voxel_spacing)
                #segmentation_data = segmentation_data[:, ::-1, :]

                # Mirror SUV vals
                SUV_vals[pet_name] = SUV_vals[pet_name][:, ::-1, :]

                

                # Upscale the PET SUV values to match the shape of the segmentation
                new_shape = (segmentation_data.shape[0], segmentation_data.shape[1], segmentation_data.shape[2])
                print("Shape of segmentation is: ", new_shape)
                print("pet_name is:", pet_name)
                print("Shape of PET suv_vals is: ", SUV_vals[pet_name].shape)

                # suv_values, segmentation_shape, pet_shape, pet_spacing, segmentation_spacing
                upscaled_suv_values = old_upscale_suv_values_3d(SUV_vals[pet_name], new_shape)
                #upscaled_suv_values = upscale_suv_values_3d(SUV_vals[pet_name], new_shape, SUV_vals[pet_name].shape, spacings[pet_name], segmentation_voxel_spacing) #spacings[pet_name], new_shape, )#upscale_suv_values_3d(SUV_vals[pet_name], new_shape)
                #upscaled_suv_values = upscale_pet_to_segmentation_with_voxel_spacing(SUV_vals[pet_name], SUV_vals[pet_name].shape, new_shape, spacings[pet_name], segmentation_voxel_spacing)
                print("finished upscaling")

                downscaled_seg = old_upscale_suv_values_3d(segmentation_data, SUV_vals[pet_name].shape)

                scroll_through_3d_array(downscaled_seg, SUV_vals[pet_name])

                # from radiomics import featureextractor

                # # Initialize the extractor
                # extractor = featureextractor.RadiomicsFeatureExtractor()

                # # Extract features
                # result = extractor.execute('pet_suv.nii.gz', seg_dir)

                # # Print extracted statistics
                # for feature_name, feature_value in result.items():
                #     print(f"{feature_name}: {feature_value}")


                #downscaled_seg = upscale_suv_values_3d(segmentation_data, SUV_vals[pet_name].shape)

                stats[pet_name] = more_optimized_suv_statistics(roi_data=segmentation_data, pet_data=upscaled_suv_values, reference=name_reference)
                del segmentation_data, upscaled_suv_values
                gc.collect()
                counter = counter + 1
    return stats

tasks = [
    "total",
    "lung_vessels",
    "body",
    "cerebral_bleed",
    "hip_implant",
    "coronary_arteries",
    "pleural_pericard_effusion",
    "head_glands_cavities",
    "head_muscles",
    "headneck_bones_vessels",
    "headneck_muscles",
    "liver_vessels",
    "oculomotor_muscles",
    "heartchambers_highres",
    "appendicular_bones",
    "tissue_types",
    "brain_structures",
    "vertebrae_body",
    "face"
]

#assumptions: organization = folder of patients with folder named PET inside with dicom
def main():

    # GUI
    layout = [
        [sg.Text("Home Directory:"), sg.Input(key="HOME_DIR"), sg.FolderBrowse()],
        [sg.Text("Segmentation Directory:"), sg.Input(key="SEGMENTATION_DIR"), sg.FolderBrowse()],
        [sg.Text("PET NIFTI Output Directory:"), sg.Input(key="NIFTI_OUTPUT_DIR"), sg.FolderBrowse()],
        [sg.Text("Task:"), sg.Combo(tasks, default_value="total", key="TASK")],
        [sg.Text("CSV Output Directory:"), sg.Input(key="CSV_OUTPUT_DIR"), sg.FolderBrowse()],
        [sg.Text("CSV Output Name:"), sg.Input(key="TEXT_INPUT", size=(40, 1))],
        [sg.Button("Run"), sg.Button("Exit")]
    ]

    window = sg.Window("Stats from Segmentations GUI", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WINDOW_CLOSED:
            break
        elif event == "Run":
            
            home_dir = values["HOME_DIR"]
            seg_dir = values["SEGMENTATION_DIR"]
            nifti_output_dir = values["NIFTI_OUTPUT_DIR"]
            csv_output_dir = values["CSV_OUTPUT_DIR"]
            task = values["TASK"]
            csv_name = values["TEXT_INPUT"]

            if not home_dir or not seg_dir or not nifti_output_dir or not task or not csv_name:
                sg.popup_error("All fields must be filled out!")
                continue
            try:
                 # Convert PET DICOMs to NIFTIs
                dicom2nifti(home_dir, nifti_output_dir)

                # Extract Stats

                SUV_vals, metadata = extractSUVs(nifti_output_dir)
                names = globals().get(task, None)
                rois = list(range(1,len(names)+1))

                print("directories:")
                print(seg_dir)
                print(nifti_output_dir)
                stats = statistics_from_rois(seg_dir, nifti_output_dir, ".nii.gz", name_reference=names)

                csv_path = os.path.join(csv_output_dir, csv_name+".csv")
                save_to_csv(stats, csv_path)
                sg.popup("Segmentation(s) completed successfully!")
            except Exception as e:
                sg.popup_error(f"An error occurred: {e}")
    
    window.close()

    # #Inputs
    # nifti_dir = "H:/pet nifti/"
    # pet_dicom_dir = "H:/test/"

    # segmentation_dir = "H:/seg/"

    # task = "total"
    # csv_name = "test"

   
if __name__ == "__main__":
    start_time = time.time()
    import multiprocessing

    # Pyinstaller fix
    multiprocessing.freeze_support()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

