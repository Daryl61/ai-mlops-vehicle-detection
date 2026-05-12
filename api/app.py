from fastapi import FastAPI, UploadFile
from pipeline import process_single_frame
from detect import detect_objects
import numpy as np
import cv2 as cv
import os

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect(file: UploadFile):
    file_read = file.file.read()
    nppar = np.frombuffer(file_read, np.uint8)
    img = cv.imdecode(nppar, cv.IMREAD_COLOR)
    detections = detect_objects(img)
    return {"detections": detections}


@app.get("/detect-live")
def detect_live():
    frame, detections = process_single_frame()
    return {"detections": detections}


@app.get("/stats")
def stats():
    raw_count = len([f for f in os.listdir("data/raw_frames") if f.endswith(".jpg")])
    labeled_count = len([f for f in os.listdir("data/labeled") if f.endswith(".jpg")])
    return {"raw_count": raw_count, "labeled_count": labeled_count}
