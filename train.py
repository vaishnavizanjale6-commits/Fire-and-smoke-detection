"""
train.py
--------
Fast CPU training for YOLO11n Fire/Smoke Detection.
"""

import argparse
import os
import shutil
import sys

DEFAULT_DATA_YAML = os.path.join("dataset", "data.yaml")
DEFAULT_BASE_MODEL = "yolo11n.pt"
DEFAULT_PROJECT = "runs"
DEFAULT_RUN_NAME = "fire_smoke"
DEFAULT_MODELS_DIR = "models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on the fire/smoke dataset."
    )

    parser.add_argument("--data", default=DEFAULT_DATA_YAML)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)

    # FAST CPU SETTINGS
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)

    return parser.parse_args()


def check_dataset(data_yaml):
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset config not found: {data_yaml}")
        sys.exit(1)


def main():
    args = parse_args()

    check_dataset(args.data)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"[INFO] Loading base model: {args.model}")

    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "workers": 0,
        "patience": 5,
        "cache": False,
    }

    print("\n[INFO] FAST CPU TRAINING")
    print("--------------------------------")
    for key, value in train_kwargs.items():
        print(f"{key}: {value}")
    print("--------------------------------\n")

    results = model.train(**train_kwargs)

    run_dir = os.path.join(
        args.project,
        "detect",
        args.project,
        args.name,
        "weights"
    )

    # Try normal Ultralytics output path too
    possible_paths = [
        os.path.join(args.project, args.name, "weights", "best.pt"),
        os.path.join(args.project, "detect", args.project, args.name, "weights", "best.pt"),
        os.path.join(args.project, "detect", args.name, "weights", "best.pt"),
    ]

    best_path = None

    for path in possible_paths:
        if os.path.exists(path):
            best_path = path
            break

    # Final fallback: search recursively
    if best_path is None:
        for root, dirs, files in os.walk(args.project):
            if "best.pt" in files:
                best_path = os.path.join(root, "best.pt")
                break

    print("\n[INFO] Training completed.")

    if best_path:
        os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)

        destination = os.path.join(
            DEFAULT_MODELS_DIR,
            "best.pt"
        )

        shutil.copy2(best_path, destination)

        print(f"[SUCCESS] best.pt copied to:")
        print(destination)

    else:
        print("[WARNING] best.pt was not found.")
        print("Search manually using:")
        print("Get-ChildItem -Recurse -Filter best.pt")


if __name__ == "__main__":
    main()