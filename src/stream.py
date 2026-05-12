import cv2 as cv
import time
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)



def stream_video():
    cap=cv.VideoCapture(config["stream"]["url"])

    while True:
        ret,frame=cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        time.sleep(config["stream"]["frame_interval"])
        frame=cv.resize(frame, (config["stream"]["width"], config["stream"]["height"]))
        yield frame
    cap.release()
