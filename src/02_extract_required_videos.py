import pandas as pd
import shutil
import zipfile
from pathlib import Path
# Paths
PROJECT_DIR= Path(r"E:\yolo11\AccidentDetection")

# Folder (extracted AIC21 train videos)
AICITY_DIR = PROJECT_DIR / "data" / "raw" / "aicity"
zip_files = list(AICITY_DIR.rglob("*.zip"))
if not zip_files:
    print("\n[ERROR] No ZIP file found")
    exit()
ZIP_FILE = zip_files[0]
print("\nZIP file found:")
print(ZIP_FILE)


# Annotation csv file
ANNOTATION_FILE= (PROJECT_DIR/"data"/"processed"/"anomalies.csv")
# Output folder path
OUTPUT_DIR= (PROJECT_DIR/"data"/"clips"/"anomaly")
# Create output folder
OUTPUT_DIR. mkdir(parents=True, exist_ok="True")

# Loading Annotations
df= pd.read_csv(ANNOTATION_FILE)
video_ids= sorted(df["video_id"].unique())

print("==========================================")
print("STEP 8A - EXTRACT REQUIRED TRAIN VIDEOS")
print("==========================================")

print("\nTotal anomaly records:", len(df))
print(f"Videos required: {len(video_ids)}")

print("\nVideo IDs:")
print(video_ids)

# Checking zip file
if not ZIP_FILE.exists():
    print("\n[ERROR] zip file not found:")
    print(ZIP_FILE)
    exit()
print("\nZIP file found:")
print(ZIP_FILE)

# Open ZIP 
with zipfile.ZipFile(ZIP_FILE, "r")as z:
    all_files= z.namelist()
    print("\nZIP opened successfully.")
    print("Searching for required videos...\n")

    extracted_count=0

    for video_id in video_ids:
        target_name= f"{video_id}.mp4"

        # Finding the train-data video inside the ZIP
        matching_files=[file for file in all_files if file.lower().endswith(f"aic21-track4-train-data/{target_name}".lower())]

        if not matching_files:
            print(f"[Not Found] {target_name}")
            continue

        zip_video_path= matching_files[0]
        output_path= OUTPUT_DIR/target_name
        print(f"[FOUND] {zip_video_path}")

        #Extract only this video
        with z.open(zip_video_path) as source:
            with open(output_path,"wb") as target:
                shutil.copyfileobj(source, target)

        extracted_count += 1
        print(f"[SAVED] {output_path}")

print(f"\nVideos extracted: {extracted_count}")
print(f"Videos requested: {len(video_ids)}")

print("\nSaved in:")
print(OUTPUT_DIR)





















