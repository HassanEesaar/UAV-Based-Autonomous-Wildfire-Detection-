"""
Evaluate YOLO model performance on the test set.

Usage:
    python scripts/evaluate.py --weights weights/yolov12n_wildfire.pt --data configs/yolo_config.yaml
"""

import argparse
from ultralytics import YOLO


def evaluate(weights: str, data: str, imgsz: int, device: str):
    model = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        split="test",
        save_json=True,
        plots=True,
        project="results/evaluation",
    )

    print("\n===== Evaluation Results =====")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print("==============================\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/yolov12n_wildfire.pt")
    parser.add_argument("--data", default="configs/yolo_config.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.imgsz, args.device)
