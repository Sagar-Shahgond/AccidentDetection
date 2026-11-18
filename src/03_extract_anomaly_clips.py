
import cv2
import pandas as pd
from pathlib import Path

# PROJECT PATH
PROJECT_DIR = Path(r"E:\yolo11\AccidentDetection")
# INPUTS
# The 17 videos extracted in Step 8A
VIDEO_DIR = (PROJECT_DIR/"data"/"clips"/"anomaly")
# Annotation csv file
ANNOTATION_FILE= (PROJECT_DIR/"data"/"processed"/"anomalies.csv")
# Output folder path
OUTPUT_DIR= (PROJECT_DIR/"data"/"clips"/"processed")
# Create output folder
OUTPUT_DIR. mkdir(parents=True, exist_ok="True")

# Loading Annotation 
df= pd.read_csv(ANNOTATION_FILE)


# PROCESS EACH ANOMALY RECORD
extracted_count=0
for index, row in df.iterrows():
    video_id= int(row["video_id"])
    start_time= float(row["start_time"])
    end_time= float(row["end_time"])

    video_path= VIDEO_DIR/f"{video_id}.mp4"
    print("\n--------------------------------------")
    print(f"Processing video: {video_id}")
    print(f"Start time: {start_time} sec")
    print(f"End time:   {end_time} sec")
    print("--------------------------------------")

    # checking if video exists
    if not video_path.exists():
        print(f"[SKIP] Video not found: {video_path}")
        continue

    # Open video
    cap= cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] could not open: {video_path}")
        continue

    # Getting video info
    fps= cap.get(cv2.CAP_PROP_FPS)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <=0:
        print("[ERROR] Invalid FPS")
        cap.release()
        continue
    
    # Converting timestamp to frame
    start_frame= int(start_time*fps)
    end_frame= int(end_time*fps)

    # Make sure end frame is inside video
    if end_frame >= total_frames:
        end_frame = total_frames - 1


    # Move to starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

    # Output filename
    output_filename= (f"video_{video_id}_{int(start_time)}_{int(end_time)}.mp4")
    output_path= OUTPUT_DIR/output_filename

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out= cv2.VideoWriter(str(output_path), fourcc, fps, (width,height))
    
    # Extract Frames
    frame_number= start_frame
    frames_written = 0
    while frame_number <= end_frame:
        success, frame= cap.read()
        if not success:
            break
        out.write(frame)
        frame_number +=1
        frames_written +=1

    cap.release()
    out.release()
    print(f"[SAVED] {output_path.name}")


# Result
if frames_written > 0:
    extracted_count += 1
    duration = frames_written / fps
    print(f"[SAVED] {output_filename}")
    print(f"Frames: {frames_written}")
    print(f"Duration: {duration:.2f} seconds")
else:
    print(f"[ERROR] No frames extracted for {output_filename}")
