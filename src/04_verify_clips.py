import cv2
from pathlib import Path 
PROJECT_DIR= Path(r"E:\yolo11\AccidentDetection")
CLIPS_DIR=(PROJECT_DIR/"data"/"clips"/"processed")
clips= sorted(CLIPS_DIR.glob("*mp4"))
print(f"Clips found: {len(clips)}")
if len(clips)==0:
    print("\n[ERROR] No clips found.")
    print("Check:")
    print(CLIPS_DIR)
    exit()

valid_clips=0
invalid_clips=0
for clip in clips:
    cap= cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {clip.name}")
        invalid_clips+=1
        continue

    frame_count= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps= cap.get(cv2.CAP_PROP_FPS)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculating duration
    if fps>0:
        duration= frame_count/fps
    else:
        duration=0
    cap.release()

    # Validity
    if frame_count>0 and fps>0:
        valid_clips+=1
        print(f"\n[VALID] {clip.name}")
        print(f"  Frames    : {frame_count}")
        print(f"  FPS       : {fps:.2f}")
        print(f"  Resolution: {width} x {height}")
        print(f"  Duration  : {duration:.2f} seconds")
    else:
        invalid_clips += 1
        print(f"\n[INVALID] {clip.name}")


print("STEP 9 COMPLETE")
print(f"\nTotal clips : {len(clips)}")
print(f"Valid clips : {valid_clips}")
print(f"Invalid clips: {invalid_clips}")
print("\nVerified folder:")
print(CLIPS_DIR)