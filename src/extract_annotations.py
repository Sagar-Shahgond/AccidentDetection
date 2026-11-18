from pathlib import Path
import zipfile
zip_path= Path(r"E:\yolo11\AccidentDetection\data\raw\aicity\AICity21-Track4-Anomaly-Detection.zip")
output_dir= Path(r"E:\yolo11\AccidentDetection\data\raw\aicity")
with zipfile.ZipFile(zip_path,"r") as z:
    for file in z.namelist():
        if "train-anomaly-results.txt" in file.lower():
            print("File Found:\n",file)
            z.extract(file, output_dir)
            print("\nAnnotation Extracted Successfully.")
            break