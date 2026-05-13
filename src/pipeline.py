from stream import stream_video
from detect import detect_objects
from collector import collect_low_confidence
import cv2 as cv
import argparse


def process_single_frame():
    frame = next(stream_video())
    detections = detect_objects(frame)
    collect_low_confidence(frame, detections)
    return frame, detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.live:
        from stream import config
        config["stream"]["frame_interval"] = 0

    count_frames = 0

    try:
        for frame in stream_video():
            detections = detect_objects(frame)
            collect_low_confidence(frame, detections)
            print(f"Detected: {len(detections)} vehicles")
            count_frames += 1

            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv.imshow("Pipeline", frame)
            if cv.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print(f"Toplam işlenen frame sayısı: {count_frames}")

    cv.destroyAllWindows()
