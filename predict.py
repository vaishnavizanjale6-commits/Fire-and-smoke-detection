"""
predict.py
----------
Command-line prediction script. Runs the trained YOLO11n fire/smoke model
on an image, a video file, or the live webcam, and saves/shows the
annotated output. This mirrors app.py's detection logic but without the
Streamlit UI, useful for quick tests or batch processing.

Usage:
    python predict.py --source path/to/image.jpg
    python predict.py --source path/to/video.mp4 --save output.mp4
    python predict.py --source webcam
    python predict.py --source webcam --threshold 0.5
"""

import argparse
import os
import sys

import cv2

from src.detection_utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DetectionError,
    build_alert_message,
    load_model,
    open_webcam,
    run_inference,
    validate_image_file,
    validate_video_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fire/smoke detection on an image, video, or webcam.")
    parser.add_argument("--source", required=True, help="Path to image/video file, or 'webcam'")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model weights (.pt)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold (0-1)")
    parser.add_argument("--save", default=None, help="Optional path to save the annotated image/video")
    parser.add_argument("--show", action="store_true", help="Display output in a window (requires a display)")
    return parser.parse_args()


def predict_image(model, path: str, threshold: float, save_path: str | None, show: bool) -> None:
    validate_image_file(path)
    frame = cv2.imread(path)
    if frame is None:
        raise DetectionError(f"Could not read image file: {path}")

    result = run_inference(model, frame, threshold)
    print(f"[RESULT] {result.status}")
    for det in result.detections:
        print(f"  - {det.label} ({det.confidence * 100:.1f}%) at box {det.box}")
    print(build_alert_message(result))

    if save_path:
        cv2.imwrite(save_path, result.annotated_frame)
        print(f"[INFO] Annotated image saved to {save_path}")

    if show:
        cv2.imshow("Fire and Smoke Detection", result.annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def predict_video(model, path: str, threshold: float, save_path: str | None, show: bool) -> None:
    validate_video_file(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise DetectionError(f"Could not open video file: {path}")
    _process_stream(model, cap, threshold, save_path, show)


def predict_webcam(model, threshold: float, save_path: str | None, show: bool) -> None:
    cap = open_webcam(0)
    _process_stream(model, cap, threshold, save_path, show, is_webcam=True)


def _process_stream(model, cap: cv2.VideoCapture, threshold: float, save_path: str | None, show: bool, is_webcam: bool = False) -> None:
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            result = run_inference(model, frame, threshold)
            if result.is_alert:
                print(f"[FRAME {frame_count}] {build_alert_message(result)}")

            if writer is not None:
                writer.write(result.annotated_frame)

            if show:
                cv2.imshow("Fire and Smoke Detection", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Annotated video saved to {save_path}")
        if show:
            cv2.destroyAllWindows()

    if is_webcam:
        print("[INFO] Webcam session ended.")
    else:
        print(f"[INFO] Processed {frame_count} frames.")


def main() -> None:
    args = parse_args()

    try:
        model = load_model(args.model)

        if args.source.lower() == "webcam":
            predict_webcam(model, args.threshold, args.save, args.show)
        elif os.path.splitext(args.source)[1].lower() in (".mp4", ".avi", ".mov", ".mkv"):
            predict_video(model, args.source, args.threshold, args.save, args.show)
        else:
            predict_image(model, args.source, args.threshold, args.save, args.show)

    except DetectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
