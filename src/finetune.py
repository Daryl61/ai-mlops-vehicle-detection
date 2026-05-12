from ultralytics import YOLO
import yaml
import os
import mlflow
import shutil

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def model_finetune():
    files = [f for f in os.listdir("data/labeled") if f.endswith(".jpg")]
    if len(files) < config["collector"]["min_frames_for_finetune"]:
        print("Yeterli veri yok.")
    else:
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name="Finetuning"):
            model = YOLO(config["detection"]["model_path"])

            model.train(data="dataset.yaml", epochs=config["finetune"]["epochs"],
                        batch=config["finetune"]["batch_size"],
                        imgsz=config["finetune"]["imgsz"])

            mlflow.log_param("epochs", config["finetune"]["epochs"])
            mlflow.log_param("batch_size", config["finetune"]["batch_size"])
            mlflow.log_param("img_size", config["finetune"]["imgsz"])

            metrics = model.val()
            mlflow.log_metric("mAP50", metrics.box.map50)
            mlflow.log_metric("mAP50-95", metrics.box.map)
            mlflow.log_metric("precision", metrics.box.precision)
            mlflow.log_metric("recall", metrics.box.recall)

            shutil.copy("runs/detect/train/weights/best.pt", "models/best.pt")
            mlflow.log_artifact("models/best.pt")
