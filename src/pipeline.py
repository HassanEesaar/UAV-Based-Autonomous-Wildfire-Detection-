"""
WildfirePipeline: end-to-end UAV wildfire detection and alert generation.

Chains: YOLOv12n detection → Moondream2 VLM description → GPT-4o mini alert
"""

import os
from pathlib import Path

from src.detection.yolo_detector import WildfireDetector
from src.vlm.moondream_analyzer import MoondreamAnalyzer
from src.llm.alert_generator import AlertGenerator


class WildfirePipeline:
    """
    Full inference pipeline from a drone frame to an emergency alert.

    Args:
        yolo_weights (str): Path to fine-tuned YOLOv12n weights.
        vlm_model (str): Moondream2 model ID or local path.
        openai_api_key (str): OpenAI API key for GPT-4o mini.
        conf_threshold (float): YOLO confidence threshold.
        device (str): Inference device.
    """

    def __init__(
        self,
        yolo_weights: str = "weights/yolov12n_wildfire.pt",
        vlm_model: str = "vikhyatk/moondream2",
        openai_api_key: str = None,
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ):
        self.detector = WildfireDetector(
            weights=yolo_weights,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.analyzer = MoondreamAnalyzer(model_id=vlm_model, device=device)
        self.alert_gen = AlertGenerator(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        )

    def run(
        self,
        image_path: str,
        location_info: dict = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run the full pipeline on a single drone frame.

        Args:
            image_path (str): Path to the input image.
            location_info (dict): Optional {"lat": ..., "lon": ..., "place_name": ...}
            verbose (bool): Print intermediate results.

        Returns:
            dict with keys:
                - detections (list[dict])
                - description (str)
                - alert (dict): situation_summary, alert_level, emergency_guidance
        """
        # Stage 1: Object Detection
        detections = self.detector.predict(image_path)
        if verbose:
            print(f"[YOLO] {len(detections)} detection(s) found.")

        if not detections:
            return {
                "detections": [],
                "description": None,
                "alert": None,
                "message": "No fire or smoke detected in this frame.",
            }

        # Stage 2: VLM Scene Description
        description = self.analyzer.describe(image_path)
        if verbose:
            print(f"[VLM] Description generated ({len(description)} chars).")

        # Stage 3: LLM Alert Generation
        alert = self.alert_gen.generate(
            detections=detections,
            descriptions=[description],
            location_info=location_info,
        )
        if verbose:
            print(f"[LLM] Alert level: {alert['alert_level'].upper()}")

        return {
            "detections": detections,
            "description": description,
            "alert": alert,
        }
