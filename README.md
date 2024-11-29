## **READ THIS FIRST BEFORE USING**

### **CREDITS**
- **TotalSegmentator Model**: [https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)  
- **William Lee**: Drexel COM  

---

## **FOLDER LAYOUT**
This zip file contains two programs:  

1. **Batch_Segmenter**:  
   Takes CT scans and generates segmentation files.  
2. **Extract_Segmentation_Stats**:  
   Takes segmentation files, matches them with PET scans, and generates SUV statistics.  

### **Required Directory Structure**:
To use these scripts, organize your scans as follows:

home_directory

   ├── patient_1/
   
      ├── PET/
      
      └── CT/
      
   ├── patient_2/
   
      ├── PET/
      
      └── CT/


- Folders **PET** and **CT** **must** be named exactly as shown.  
- These folders must contain all DICOM files for the corresponding PET and CT scans.  
- The root folder, referred to as `home_directory`, contains all patient folders and their scans.

---

## **STEPS TO USE**

### **1) Organize Your Scan Database**
Ensure the database matches the directory structure outlined above.

---

### **2) Run the `Batch_Segmenter` Script**
- **Step 2a**: Select the task (structures to segment).  
  - Full list of tasks: [TotalSegmentator Subtasks](https://github.com/wasserth/TotalSegmentator#subtasks).  
- **Step 2b**: Select the **Home Directory** (organized scan database).  
- **Step 2c**: Select the **Output Directory** (should be an empty folder for the segmentation files).  
- **Step 2d**: Choose Low Resolution (optional):  
  - For faster preliminary data, select **low resolution** segmentation.  

---

### **3) Run the `Extract_Segmentation_Stats` Script**
- **Step 3a**: Select the **Home Directory** (organized scan database).  
- **Step 3b**: Select the **Segmentation Directory** (output folder from the `Batch_Segmenter` script).  
- **Step 3c**: Select the **PET NIFTI Output Directory** (an empty folder where converted PET scans will be saved as NIFTI files).  
- **Step 3d**: Select the **Task**:  
  - Choose the same task used during segmentation (e.g., "total").  
- **Step 3e**: Select the **CSV Output Directory**:  
  - Choose the folder where the CSV statistics file will be saved.  
- **Step 3f**: Set the **CSV Output Name**:  
  - Provide a name for the output CSV file.

---

## **NOTES**
- If you lack a dedicated GPU, the process may run much slower.  
- Large datasets may take considerable time (e.g., over 100 hours).  
- If the script appears unresponsive, it is likely still running. Check the terminal window for updates.  
- For questions or bug reports, contact me directly.  

---
