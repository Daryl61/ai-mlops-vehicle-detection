import json
import cv2 as cv
import time
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def collect_low_confidence(frame, detections):
    low_conf = [det for det in detections if det["confidence"] < config["detection"]["confidence_threshold"]]

    if low_conf:
        timestamp = int(time.time())
        save_path = config["collector"]["save_path"]
        cv.imwrite(f"{save_path}/{timestamp}.jpg", frame)

        with open(f"{save_path}/{timestamp}.json", "w") as f:
            json.dump({"timestamp": timestamp, "detections": low_conf}, f)
