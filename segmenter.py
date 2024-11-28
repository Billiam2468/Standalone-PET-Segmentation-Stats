import os
import subprocess

def runBash(command):
    #Execute bash command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print the output
    if stdout:
        print("Output:")
        print(stdout.decode())

def segment(DICOM, output_dir, segmentName, task):
    # output = DICOM.split('\\')
    # output_str = output[3] + "_" + output[5]
    # os.makedirs("E:/Psoriasis/AI Segmentations/" + output_str)
    # d2n.convert_directory(DICOM, "E:/Psoriasis/AI Segmentations/" + output_str, compression=True)
    command = f'TotalSegmentator -i "{DICOM}" -o "{output_dir}/{segmentName}" --ml --force_split -ta {task}'
    #command = f'moosez -d {DICOM} -m {task}'
    runBash(command)

if __name__ == "__main__":
    home_dir = "F:/UC Davis Melanoma/"
    output_dir = "H:/Melanoma Project/AI Segmentations/"
    task = "total"
    with os.scandir(home_dir) as patients:
        for patient in patients:
            if patient.is_dir():
                scan_dir = os.path.join(home_dir, patient, "CT")
                seg_name = patient.name + ".nii.gz"
                segment(DICOM=scan_dir, output_dir=output_dir, segmentName=seg_name, task=task)


def main():
    # These 4 are the inputs
    fast = True
    task = "total"
    home_dir = "H:/test/"
    output_dir = "H:/Temp/"

    with os.scandir(home_dir) as patients:
        for patient in patients:
            if patient.is_dir():
                scan_dir = os.path.join(home_dir, patient, "CT")
                seg_name = patient.name + ".nii.gz"
                output_path = output_dir + seg_name
                totalsegmentator(scan_dir, output_path, ml=True, fast=fast, task=task, force_split=True, device="gpu")
