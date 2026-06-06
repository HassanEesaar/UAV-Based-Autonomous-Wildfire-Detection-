"""
Download and prepare the wildfire + smoke dataset from Roboflow.

Usage:
    python scripts/prepare_dataset.py --api-key YOUR_ROBOFLOW_API_KEY
"""

import argparse
import os


def download_dataset(api_key: str, workspace: str, project: str, version: int, output_dir: str):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=output_dir)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def verify_dataset(dataset_dir: str):
    splits = ["train", "val", "test"]
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        n_imgs = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        n_lbls = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
        print(f"  {split:10s}: {n_imgs:4d} images, {n_lbls:4d} labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--workspace", default="wildfire-detection", help="Roboflow workspace")
    parser.add_argument("--project", default="smoke-wildfire", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--output", default="data/wildfire_dataset")
    args = parser.parse_args()

    print("Downloading dataset from Roboflow...")
    loc = download_dataset(args.api_key, args.workspace, args.project, args.version, args.output)

    print("\nDataset summary:")
    verify_dataset(loc)
