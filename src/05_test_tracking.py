import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR= Path(r"E:\yolo11\AccidentDetection")
INPUT_VIDEO= (PROJECT_DIR/"data"/"clips"/"processed"/"video_2_587_894.mp4")
OUTPUT_DIR= (PROJECT_DIR/"outputs"/"videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO= (OUTPUT_DIR/"tracking_test.mp4")

# Starting with yolo + bytetrack
print(f"\nInput video: {INPUT_VIDEO}")
if not INPUT_VIDEO.exists():
    print("\n[ERROR] Input video not found.")
    exit()

# Loading model
model= YOLO("yolo11n.pt")
print(f"\nYolo model loaded successfully.")

# Opening video
cap= cv2.VideoCapture(str(INPUT_VIDEO))
if not cap.isOpened():
    print("\n[ERROR] Could not open input video.")
    exit()

fps= cap.get(cv2.CAP_PROP_FPS)
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("\nVideo information:")
print(f"FPS        : {fps}")
print(f"Resolution : {width} x {height}")
print(f"Total frames: {total_frames}")

# Limit test to first 30 seconds
max_frames= int(30 * fps)
print(f"\nTesting first 30 seconds...")
print(f"Maximum frames: {max_frames}")

# Video Writer
fourcc= cv2.VideoWriter_fourcc(*"mp4v")
out= cv2.VideoWriter(str(OUTPUT_VIDEO),fourcc, fps, (width,height))

# Tracking 
frame_number= 0
unique_track_ids= set()

print("\nStarting YOLO + ByteTrack...")
print("Please wait...\n")

while frame_number < max_frames:
    success, frame= cap.read()
    if not success:
        break
    frame_number+=1
    results= model.track(frame, persist= True, tracker="bytetrack.yaml", conf=0.3, verbose= False)
    result= results[0]

    # Getting track id
    if result.boxes.id is not None:
        track_ids= (result.boxes.id.cpu().numpy().astype(int))

    for track_id in track_ids:
        unique_track_ids.add(track_id)

    # Draw YOLO + Tracking reults
    annotated_frame= result.plot()
    out.write(annotated_frame)

    # Progress
    if frame_number % 100 ==0:
        current_time= frame_number/fps
        print(f"Processed: "f"{current_time:.1f}s | "f"Unique tracks: {len(unique_track_ids)}")

cap.release()
out.release()

print("STEP 10 COMPLETE")
print(f"\nFrames processed : {frame_number}")
print(f"Duration tested  : {frame_number / fps:.2f} seconds")
print(f"Unique track IDs : {len(unique_track_ids)}")

print("\nOutput video:")
print(OUTPUT_VIDEO)