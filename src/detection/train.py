"""
Fine-tune YOLOv12n on the wildfire + smoke dataset.

Usage:
    python src/detection/train.py --data configs/yolo_config.yaml --epochs 100
"""

import argparse
from ultralytics import YOLO


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, device: str):
    model = YOLO("yolov12n.pt")  # load pretrained nano weights

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="results/training",
        name="yolov12n_wildfire",
        patience=20,
        save=True,
        plots=True,
    )
    print(f"Training complete. Best weights at: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv12n for wildfire detection")
    parser.add_argument("--data", type=str, default="configs/yolo_config.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.device)
