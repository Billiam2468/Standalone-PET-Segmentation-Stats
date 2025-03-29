import os
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk

def load_dicom_series(dicom_dir):
    """Load a DICOM series from a directory."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    return image

dicom_path = "/Users/williamlee/Documents/Example/Home database/31/CT/IM-0001-0001-0001.dcm"
ds = pydicom.dcmread(dicom_path)
print(ds.RescaleIntercept)
print(ds.RescaleSlope)

# HU = pixel value * slope + intercept

# Takes the relevant metadata from the DICOM file including the directory it came from and moves it into the metadata of the NIFTI file
def move_DICOM_Metadata_To_NIFTI(dicom_path, nifti_path, dicom_folder):
    # Adding a try-except block in cases where some metadata isn't available
    try:
        ds = pydicom.dcmread(dicom_path)
        seriesTime = ds.SeriesTime if hasattr(ds, "SeriesTime") else "N/A"
        injectionTime = (
            ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            if hasattr(ds, "RadiopharmaceuticalInformationSequence") and len(ds.RadiopharmaceuticalInformationSequence) > 0
            else "N/A"
        )
        halfLife = (
            ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
            if hasattr(ds, "RadiopharmaceuticalInformationSequence") and len(ds.RadiopharmaceuticalInformationSequence) > 0
            else "N/A"
        )
        injectedDose = (
            ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            if hasattr(ds, "RadiopharmaceuticalInformationSequence") and len(ds.RadiopharmaceuticalInformationSequence) > 0
            else "N/A"
        )
        patientWeight = ds.PatientWeight if hasattr(ds, "PatientWeight") else "N/A"
    except Exception as e:
        # Placeholder values if an error occurs
        print(f"Error reading DICOM metadata. Some metadata wasn't available in the PET: {e}")
        seriesTime = "N/A"
        injectionTime = "N/A"
        halfLife = "N/A"
        injectedDose = "N/A"
        patientWeight = "N/A"

    nifti_file = nib.load(nifti_path)

    nifti_file.header['descrip'] = f"{seriesTime}x{injectionTime}x{halfLife}x{injectedDose}x{patientWeight}"
    directory_binary = dicom_folder.encode('utf-8')  # Convert to bytes
    nifti_file.header.extensions.append(nib.nifti1.Nifti1Extension(code=6, content=directory_binary))

    nib.save(nifti_file, nifti_path)