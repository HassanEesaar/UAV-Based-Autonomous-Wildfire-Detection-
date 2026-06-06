"""
Run wildfire detection on a video file or live stream.

Usage:
    python scripts/run_video.py \\
        --source path/to/video.mp4 \\
        --weights weights/yolov12n_wildfire.pt \\
        --output results/output.mp4 \\
        --generate-alerts
"""

import argparse
import cv2
import os
from src.detection.yolo_detector import WildfireDetector
from src.pipeline import WildfirePipeline


def run_detection_only(source, weights, output, conf):
    detector = WildfireDetector(weights=weights, conf_threshold=conf)
    cap = cv2.VideoCapture(source)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tmp = f"/tmp/frame_{frame_count}.jpg"
        cv2.imwrite(tmp, frame)
        annotated = detector.annotate(tmp)
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to: {output}")


def run_with_alerts(source, weights, openai_key, output, conf):
    pipeline = WildfirePipeline(
        yolo_weights=weights,
        openai_api_key=openai_key,
        conf_threshold=conf,
    )
    cap = cv2.VideoCapture(source)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every 30 frames for alert generation (avoid API rate limits)
        if frame_count % 30 == 0:
            tmp = f"/tmp/frame_{frame_count}.jpg"
            cv2.imwrite(tmp, frame)
            result = pipeline.run(tmp, verbose=True)

            if result.get("alert"):
                print(f"\n--- Frame {frame_count} Alert ---")
                print(result["alert"]["situation_summary"])
                print(result["alert"]["emergency_guidance"])

        frame_count += 1

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--weights", type=str, default="weights/yolov12n_wildfire.pt")
    parser.add_argument("--output", type=str, default="results/output.mp4")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--generate-alerts", action="store_true")
    parser.add_argument("--openai-key", type=str, default=None)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.generate_alerts:
        key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        run_with_alerts(args.source, args.weights, key, args.output, args.conf)
    else:
        run_detection_only(args.source, args.weights, args.output, args.conf)
