"""
detection_utils.py
-------------------
Reusable backend module for the Fire and Smoke Detection project.

This module centralizes:
    - Model loading (with error handling)              -> FR-12
    - Image / video / webcam input validation           -> FR-02, FR-03, FR-13
    - Preprocessing                                       -> FR-04
    - YOLO11n inference                                    -> FR-05, FR-06
    - Confidence threshold checking                        -> FR-08
    - Bounding box drawing                                  -> FR-05, FR-06, FR-10
    - Alert / status message generation                     -> FR-07, FR-09
    - Custom, descriptive error handling                    -> FR-13

Keeping this logic in one module (instead of duplicating it in app.py,
predict.py and evaluate.py) satisfies the TRD's requirement that the
"backend should be organized into reusable modules for easier
maintenance."
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("fire_smoke_detection")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ALLOWED_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
ALLOWED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_MODEL_PATH = os.path.join("models", "best.pt")

# Class names used by the fire/smoke YOLO model.
# NOTE: this must match the order used in dataset/data.yaml / during training.
CLASS_NAMES = {0: "fire", 1: "smoke"}

FIRE_COLOR = (0, 0, 255)     # red   (BGR) for fire boxes
SMOKE_COLOR = (128, 128, 128)  # gray  (BGR) for smoke boxes
DEFAULT_COLOR = (0, 255, 0)   # green fallback


# --------------------------------------------------------------------------- #
# Custom exceptions -> FR-13 / TRD section 15 (clear, specific error messages)
# --------------------------------------------------------------------------- #

class DetectionError(Exception):
    """Base class for all handled errors in this project."""


class ModelLoadError(DetectionError):
    """Raised when the YOLO model file is missing or fails to load."""


class InvalidInputError(DetectionError):
    """Raised when an uploaded/selected file is not a supported format or is corrupted."""


class WebcamNotAvailableError(DetectionError):
    """Raised when the webcam cannot be opened."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class Detection:
    """A single fire/smoke detection result."""
    class_id: int
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class FrameResult:
    """All detections + summary status for one image / frame."""
    detections: List[Detection] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None

    @property
    def status(self) -> str:
        """FR-07 / FR-11: overall status text for the processed frame."""
        if not self.detections:
            return "No Fire or Smoke Detected"
        labels = sorted({d.label for d in self.detections})
        return " & ".join(l.capitalize() for l in labels) + " Detected"

    @property
    def is_alert(self) -> bool:
        """FR-09: whether a warning should be raised."""
        return len(self.detections) > 0

    @property
    def max_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)


# --------------------------------------------------------------------------- #
# Validation helpers -> FR-02, FR-03, FR-13
# --------------------------------------------------------------------------- #

def validate_image_file(filename: str) -> None:
    """Raise InvalidInputError if the filename does not have an allowed image extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise InvalidInputError(
            f"Unsupported image format '{ext}'. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTS)}"
        )


def validate_video_file(filename: str) -> None:
    """Raise InvalidInputError if the filename does not have an allowed video extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise InvalidInputError(
            f"Unsupported video format '{ext}'. Allowed formats: {', '.join(ALLOWED_VIDEO_EXTS)}"
        )


def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes (e.g. from a Streamlit file_uploader) into an OpenCV BGR image.
    Raises InvalidInputError if the bytes cannot be decoded (corrupted file) -> FR-13.
    """
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidInputError("The uploaded file could not be read. It may be corrupted.")
    return image


# --------------------------------------------------------------------------- #
# Model loading -> FR-12
# --------------------------------------------------------------------------- #

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained YOLO11n model.

    Returns an ultralytics.YOLO instance.
    Raises ModelLoadError with a clear message if the file is missing or
    fails to load (FR-12, FR-13).
    """
    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"Model file not found at '{model_path}'. "
            "Train the model with train.py or place a trained 'best.pt' "
            "inside the models/ folder before running the application."
        )

    try:
        from ultralytics import YOLO  # imported here so the whole project can be
    except ImportError as exc:                     # inspected even without ultralytics installed
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        model = YOLO(model_path)
    except Exception as exc:  # noqa: BLE001 - surface any load failure as a clear app error
        raise ModelLoadError(f"Failed to load YOLO model from '{model_path}': {exc}") from exc

    logger.info("Model loaded successfully from %s", model_path)
    return model


# --------------------------------------------------------------------------- #
# Preprocessing -> FR-04
# --------------------------------------------------------------------------- #

def preprocess_frame(frame: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
    """
    Basic preprocessing before inference.
    YOLO handles its own internal resizing/normalization, but this keeps a
    dedicated preprocessing step (as required by FR-04 / TRD section 5),
    e.g. for optionally resizing very large frames to speed up inference.
    """
    if frame is None:
        raise InvalidInputError("Received an empty frame for preprocessing.")

    if target_size:
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


# --------------------------------------------------------------------------- #
# Inference + drawing -> FR-05, FR-06, FR-08, FR-10, FR-11
# --------------------------------------------------------------------------- #

def run_inference(model, frame: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> FrameResult:
    """
    Run YOLO11n inference on a single frame, filter by confidence threshold,
    draw bounding boxes, and return a FrameResult with detections and the
    annotated frame.
    """
    frame = preprocess_frame(frame)

    try:
        results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
    except Exception as exc:  # noqa: BLE001
        raise DetectionError(f"Model inference failed: {exc}") from exc

    annotated = frame.copy()
    detections: List[Detection] = []

    if results:
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", CLASS_NAMES)

        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue  # FR-08: enforce configurable threshold

                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(
                    Detection(class_id=cls_id, label=label, confidence=conf, box=(x1, y1, x2, y2))
                )
                _draw_box(annotated, x1, y1, x2, y2, label, conf)

    return FrameResult(detections=detections, annotated_frame=annotated)


def _draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, conf: float) -> None:
    """Draw a single labeled bounding box with a confidence score onto the frame."""
    color = FIRE_COLOR if label.lower() == "fire" else SMOKE_COLOR if label.lower() == "smoke" else DEFAULT_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label.upper()} {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# --------------------------------------------------------------------------- #
# Alerts -> FR-07, FR-09
# --------------------------------------------------------------------------- #

def build_alert_message(frame_result: FrameResult) -> str:
    """Build a human-readable warning/status message for the UI (FR-07, FR-09)."""
    if not frame_result.is_alert:
        return "✅ No Fire or Smoke Detected."
    labels = sorted({d.label.capitalize() for d in frame_result.detections})
    return f"🚨 WARNING: {' & '.join(labels)} Detected! (confidence up to {frame_result.max_confidence * 100:.1f}%)"


# --------------------------------------------------------------------------- #
# Webcam helper -> FR-03, FR-13
# --------------------------------------------------------------------------- #

def open_webcam(index: int = 0) -> cv2.VideoCapture:
    """Open the default webcam or raise WebcamNotAvailableError."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise WebcamNotAvailableError(
            "Could not access the webcam. Check that it is connected and not used by another application."
        )
    return cap
