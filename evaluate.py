"""
evaluate.py
-----------
Evaluates the trained YOLO11n fire/smoke model using standard object
detection metrics: Precision, Recall, F1-score, mAP@50, mAP@50-95, and a
confusion matrix, as required by FRD/TRD section "Evaluation Metrics".

Usage:
    python evaluate.py
    python evaluate.py --model models/best.pt --data dataset/data.yaml --split test
"""

import argparse
import os
import sys

DEFAULT_MODEL = os.path.join("models", "best.pt")
DEFAULT_DATA_YAML = os.path.join("dataset", "data.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fire/smoke YOLO11n model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for evaluation")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold used during evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("        Train a model first with train.py, or place a trained best.pt inside models/.")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset config not found: {args.data}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] The 'ultralytics' package is not installed.")
        print("        Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Running evaluation on '{args.split}' split...")
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, conf=args.conf)

    print("\n========== Evaluation Results ==========")
    try:
        precision = metrics.box.mp        # mean precision across classes
        recall = metrics.box.mr           # mean recall across classes
        map50 = metrics.box.map50         # mAP@0.50
        map50_95 = metrics.box.map        # mAP@0.50:0.95
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1-score       : {f1:.4f}")
        print(f"mAP@50         : {map50:.4f}")
        print(f"mAP@50-95      : {map50_95:.4f}")
    except AttributeError:
        print("[WARN] Could not parse standard metric fields; printing raw results object below.")
        print(metrics)

    print("\n[INFO] A confusion matrix and PR-curve plots are saved automatically")
    print("       by ultralytics inside the 'runs/detect/val*' directory.")
    print("=========================================")


if __name__ == "__main__":
    main()
