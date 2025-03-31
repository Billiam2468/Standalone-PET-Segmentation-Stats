import os
import nibabel as nib
import numpy as np
import pydicom
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

class ScrollableSlices:
    def __init__(self, ct_array, seg_array):
        self.ct_array = ct_array
        self.seg_array = seg_array
        self.index = ct_array.shape[0] // 2  # Start in the middle

        self.fig, self.axs = plt.subplots(1, 2)
        self.ct_im = self.axs[0].imshow(self.ct_array[self.index], cmap='gray')
        self.seg_im = self.axs[1].imshow(self.seg_array[self.index], cmap='jet', alpha=0.5)
        self.fig.suptitle(f"Slice {self.index}")

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()

    def on_scroll(self, event):
        """Scroll through slices with mouse wheel."""
        if event.step > 0:
            self.index = min(self.index + 1, self.ct_array.shape[0] - 1)
        else:
            self.index = max(self.index - 1, 0)

        self.update_display()

    def update_display(self):
        """Update the displayed slices."""
        self.ct_im.set_array(self.ct_array[self.index])
        self.seg_im.set_array(self.seg_array[self.index])
        self.fig.suptitle(f"Slice {self.index}")
        self.fig.canvas.draw_idle()

class OverlayViewer:
    def __init__(self, ct_array, seg_array):
        self.ct_array = ct_array
        self.seg_array = seg_array
        self.index = ct_array.shape[0] // 2

        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.ct_array[self.index], cmap='gray')
        self.seg = self.ax.imshow(self.seg_array[self.index], cmap='jet', alpha=0.5)
        self.fig.suptitle(f"Slice {self.index}")

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()

    def on_scroll(self, event):
        """Scroll through slices."""
        if event.step > 0:
            self.index = min(self.index + 1, self.ct_array.shape[0] - 1)
        else:
            self.index = max(self.index - 1, 0)

        self.update_display()

    def update_display(self):
        """Update display for new slice."""
        self.img.set_array(self.ct_array[self.index])
        self.seg.set_array(self.seg_array[self.index])
        self.fig.suptitle(f"Slice {self.index}")
        self.fig.canvas.draw_idle()



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


def more_optimized_suv_statistics(roi_data, ct_data, slope, intercept, reference):
    print("in suv stats")
    # Flatten arrays for faster processing
    flat_roi_data = roi_data.ravel()
    flat_pet_data = ct_data.ravel()

    # Filter out background
    mask = flat_roi_data > 0
    flat_roi_data = flat_roi_data[mask]
    flat_pet_data = flat_pet_data[mask]

    # Unique ROI values
    unique_rois = np.unique(flat_roi_data)

    suv_stats = {}
    for roi in unique_rois:
        suv_values = flat_pet_data[flat_roi_data == roi]

        #suv_values = suv_values * slope + intercept

        if suv_values.size > 0:
            suv_stats[reference[int(roi)]] = {
                'mean': np.mean(suv_values),
                'max': np.max(suv_values),
                'min': np.min(suv_values),
                'std_dev': np.std(suv_values),
                'median': np.median(suv_values),
                'num_val': suv_values.size,
            }

    return suv_stats

dicom_path = "/Users/williamlee/Documents/Example/Home database/31/CT/IM-0001-0001-0001.dcm"
ds = pydicom.dcmread(dicom_path)
slope = ds.RescaleSlope
intercept = ds.RescaleIntercept
print(ds.RescaleIntercept)
print(ds.RescaleSlope)

dicom_image = load_dicom_series("/Users/williamlee/Documents/Example/Home database/31/CT/")
segmentation_image = sitk.ReadImage("/Users/williamlee/Documents/Example/Segmentations/31.nii.gz")




# # Attempting a resample
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(dicom_image)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampled_segmentation = resampler.Execute(segmentation_image)

ct_array = sitk.GetArrayFromImage(dicom_image)
seg_array = sitk.GetArrayFromImage(resampled_segmentation)
#seg_array = sitk.GetArrayFromImage(segmentation_image)
# print(ct_array.shape)
# print(seg_array.shape)




import numpy as np
import matplotlib.pyplot as plt


#ScrollableSlices(ct_array, seg_array)
OverlayViewer(ct_array, seg_array)





stats = more_optimized_suv_statistics(seg_array, ct_array, slope, intercept, total)

print(stats['spleen'])

# HU = pixel value * slope + intercept
# Will assume that the slope and intercept is consistent for each slice of the CT scan

# Steps for HU calculation
# Keep track of: CT directory, rescale intercept, slope (assume these two are consistent)
# While segmentation stats are being calculated, calculate HU stats as well
