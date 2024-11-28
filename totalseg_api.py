from totalsegmentator.python_api import totalsegmentator
import os
import PySimpleGUI as sg

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
    "oculomotor_muscles"
]


def main():
    # Define the layout of the GUI
    layout = [

        [sg.Text("Task:"), sg.Combo(tasks, default_value="total", key="TASK")],
        [sg.Text("Home Directory:"), sg.Input(key="HOME_DIR"), sg.FolderBrowse()],
        [sg.Text("Output Directory:"), sg.Input(key="OUTPUT_DIR"), sg.FolderBrowse()],
        [sg.Text("Fast:"), sg.Checkbox("", default=True, key="FAST")],
        #[sg.Text("Please provide credit to:", font=("Helvetica", 8))],
        [sg.Text("Please provide credit to: TotalSegmentator (https://github.com/wasserth/TotalSegmentator) and William Lee (Drexel COM)", font=("Helvetica", 8))],
        #[sg.Text("William Lee (DUCOM)", font=("Helvetica", 8))],
        [sg.Button("Run"), sg.Button("Exit")]
    ]

    # Create the window
    window = sg.Window("TotalSegmentator GUI", layout)

    while True:
        event, values = window.read()

        if event == "Exit" or event == sg.WINDOW_CLOSED:
            break
        elif event == "Run":
            # Extract inputs
            fast = values["FAST"]
            task = values["TASK"]
            home_dir = values["HOME_DIR"]
            output_dir = values["OUTPUT_DIR"]

            if not home_dir or not output_dir or not task:
                sg.popup_error("All fields must be filled out!")
                continue

            # Run the segmentation logic
            try:
                with os.scandir(home_dir) as patients:
                    for patient in patients:
                        if patient.is_dir():
                            scan_dir = os.path.join(home_dir, patient.name, "CT")
                            seg_name = patient.name + ".nii.gz"
                            output_path = os.path.join(output_dir, seg_name)
                            totalsegmentator(
                                scan_dir, output_path, ml=True, fast=fast, task=task,
                                force_split=True, device="gpu"
                            )
                sg.popup("Segmentation completed successfully!")
            except Exception as e:
                sg.popup_error(f"An error occurred: {e}")

    window.close()



if __name__ == "__main__":
    main()