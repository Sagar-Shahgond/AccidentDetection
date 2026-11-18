import zipfile
from pathlib import Path
zip_path= Path(r"E:\yolo11\AccidentDetection\data\raw\aicity\AICity21-Track4-Anomaly-Detection.zip")
with zipfile.ZipFile(zip_path, "r") as z:
    files= z.namelist()
    print("****** Anomaly File *******")
    for file in files:
        if "anomaly" in file.lower():
            print(file)
    print("\n****** Train Data *******")
    train_files=[file for file in files if "train-data" in file.lower()]
    print(f"Number of train files: {len(train_files)}")
    for file in train_files[:20]:
        print(file)