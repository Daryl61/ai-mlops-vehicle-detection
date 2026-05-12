import os
import json
import cv2 as cv
import shutil
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

labeled_path = "data/labeled"

files = [f for f in os.listdir(config["collector"]["save_path"]) if f.endswith(".jpg")]

for file in files:
    frame = cv.imread(f"{config['collector']['save_path']}/{file}")
    h, w = frame.shape[:2]

    json_file = file.replace(".jpg", ".json")
    with open(f"{config['collector']['save_path']}/{json_file}", "r") as f:
        metadata = json.load(f)

    for det in metadata["detections"]:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    frame = cv.resize(frame, (config["stream"]["width"], config["stream"]["height"]))
    cv.imshow("Etiketle", frame)
    key = cv.waitKey(0)

    if key == ord('y'):
        label_file = file.replace(".jpg", ".txt")
        with open(f"{labeled_path}/{label_file}", "w") as lf:
            for det in metadata["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lf.write(f"0 {x_center} {y_center} {bw} {bh}\n")
        shutil.copy(f"{config['collector']['save_path']}/{file}", f"{labeled_path}/{file}")

    elif key == ord('n'):
        continue
    elif key == ord('q'):
        break

cv.destroyAllWindows()
