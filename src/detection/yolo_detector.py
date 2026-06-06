"""
WildfireDetector: YOLOv12n-based smoke and wildfire detection.
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


CLASS_NAMES = {0: "smokes", 1: "wildfire"}
COLORS = {
    "smokes": (255, 100, 0),     # Blue (BGR)
    "wildfire": (0, 255, 255),   # Cyan (BGR)
}


class WildfireDetector:
    """
    Wraps YOLOv12n for smoke and wildfire detection.

    Args:
        weights (str): Path to fine-tuned YOLO weights (.pt file).
        conf_threshold (float): Minimum confidence for a detection to be kept.
        device (str): 'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        weights: str = "weights/yolov12n_wildfire.pt",
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ):
        self.model = YOLO(weights)
        self.model.to(device)
        self.conf_threshold = conf_threshold

    def predict(self, source, save: bool = False, save_dir: str = "results/"):
        """
        Run inference on an image, video, or directory.

        Args:
            source: Path to image/video/directory, or a URL.
            save (bool): Whether to save annotated output.
            save_dir (str): Where to save annotated results.

        Returns:
            list[dict]: Detections with keys: class_name, confidence, bbox (xyxy).
        """
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            save=save,
            project=save_dir,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                detections.append(
                    {
                        "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(float(box.conf.item()), 3),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )
        return detections

    def annotate(self, image_path: str) -> np.ndarray:
        """
        Draw bounding boxes on an image and return the annotated frame.

        Args:
            image_path (str): Path to input image.

        Returns:
            np.ndarray: Annotated image (BGR).
        """
        detections = self.predict(image_path)
        img = cv2.imread(image_path)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = COLORS.get(det["class_name"], (200, 200, 200))
            label = f"{det['class_name']} {det['confidence']:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        return img

    def has_fire_or_smoke(self, source) -> bool:
        """Returns True if any detection is found above confidence threshold."""
        return len(self.predict(source)) > 0
