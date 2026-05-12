import mlflow
import yaml
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
import cv2 as cv
import os 
import json


def monitor_training():
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    all_confidences = []
    files=[f for f in os.listdir(("data/raw_frames")) if f.endswith(".json")]
    for file in files:
        with open(f"data/raw_frames/{file}", "r") as f:
            metadata = json.load(f)
        
        for det in metadata["detections"]:
            all_confidences.append(det["confidence"])


    avg_confidence = sum(all_confidences) / len(all_confidences)
    mlflow.log_metric("average_confidence", avg_confidence)
    mlflow.log_metric("total_raw_frames", len(files))
    mlflow.log_metric("total_detections", len(all_confidences))

