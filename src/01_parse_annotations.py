import pandas as pd
from pathlib import Path

annotation_file= Path(r"E:\yolo11\AccidentDetection\data\raw\aicity\AIC21-Track4-Anomaly-Detection\train-anomaly-results.txt")
output_file= Path(r"E:\yolo11\AccidentDetection\data\processed\anomalies.csv")
df= pd.read_csv(annotation_file, sep=r"\s+", header=None, names=["video_id", "start_time", "end_time"])
print("Dataset Summary")
print(f"\nTotal anomaly records: {len(df)}")
print(f"\nFirst 10 records: \n{df.head(10)}")
print(f"\nUnique videos containing anomalies: {df["video_id"].unique()}")
df.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")