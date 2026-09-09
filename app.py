"""
app.py
------
Streamlit user interface for the Fire and Smoke Detection project.

Implements:
    FR-01 Application Interface (image / video / webcam tabs)
    FR-02 Image Upload (JPG/JPEG/PNG + validation)
    FR-03 Video/Webcam Input
    FR-04 Preprocessing (via src/detection_utils.py)
    FR-05 / FR-06 Fire & Smoke Detection with bounding boxes
    FR-07 Normal-condition message
    FR-08 Confidence score + configurable threshold
    FR-09 Alert generation
    FR-10 Multiple detections per frame
    FR-11 Result display (input, output, class, confidence, status, alert)
    FR-12 Model loading with error reporting
    FR-13 Error handling for bad files / missing model / webcam issues

Run with:
    streamlit run app.py
"""

import time

import cv2
import numpy as np
import streamlit as st

from src.detection_utils import (
    ALLOWED_IMAGE_EXTS,
    ALLOWED_VIDEO_EXTS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DetectionError,
    InvalidInputError,
    ModelLoadError,
    WebcamNotAvailableError,
    build_alert_message,
    decode_image_bytes,
    load_model,
    open_webcam,
    run_inference,
)

st.set_page_config(page_title="Fire and Smoke Detection", page_icon="🔥", layout="wide")


# --------------------------------------------------------------------------- #
# Cached model loading -> FR-12
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading YOLO11n model...")
def get_model(model_path: str):
    return load_model(model_path)


def show_result(result, container) -> None:
    """FR-11: Display the annotated media, detection table, and alert message."""
    container.image(
        cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True,
    )

    if result.is_alert:
        container.error(build_alert_message(result))
    else:
        container.success(build_alert_message(result))

    if result.detections:
        table_rows = [
            {
                "Class": d.label.capitalize(),
                "Confidence": f"{d.confidence * 100:.1f}%",
                "Bounding Box (x1, y1, x2, y2)": str(d.box),
            }
            for d in result.detections
        ]
        container.table(table_rows)
    else:
        container.write(f"**Status:** {result.status}")


def main() -> None:
    st.title("🔥 Fire and Smoke Detection Using Machine Learning")
    st.caption("YOLO11n-based fire and smoke detector — image, video, and webcam support.")

    # ----------------------------- Sidebar (FR-08) ----------------------------- #
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", min_value=0.05, max_value=0.95,
        value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Detection classes:** Fire 🔴, Smoke ⚪\n\n"
        "Place a trained `best.pt` inside the `models/` folder "
        "(see `train.py`) before running detection."
    )

    # ----------------------------- Load model (FR-12/13) ----------------------------- #
    try:
        model = get_model(model_path)
    except ModelLoadError as exc:
        st.error(f"❌ Model loading failed: {exc}")
        st.stop()

    tab_image, tab_video, tab_webcam = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Webcam"])

    # ================================ IMAGE TAB (FR-02) ================================ #
    with tab_image:
        st.subheader("Image Detection")
        uploaded_image = st.file_uploader(
            "Upload an image", type=[ext.strip(".") for ext in ALLOWED_IMAGE_EXTS], key="image_uploader"
        )

        if uploaded_image is not None:
            try:
                frame = decode_image_bytes(uploaded_image.read())
                col1, col2 = st.columns(2)
                col1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

                with st.spinner("Running detection..."):
                    result = run_inference(model, frame, confidence_threshold)

                col2.markdown("**Detection Result**")
                show_result(result, col2)

            except InvalidInputError as exc:
                st.error(f"❌ {exc}")
            except DetectionError as exc:
                st.error(f"❌ Detection failed: {exc}")

    # ================================ VIDEO TAB (FR-03) ================================ #
    with tab_video:
        st.subheader("Video Detection")
        uploaded_video = st.file_uploader(
            "Upload a video", type=[ext.strip(".") for ext in ALLOWED_VIDEO_EXTS], key="video_uploader"
        )
        frame_skip = st.slider("Process every Nth frame (higher = faster)", 1, 10, 2)

        if uploaded_video is not None:
            try:
                temp_path = f"_temp_{int(time.time())}_{uploaded_video.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_video.read())

                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    raise InvalidInputError("The uploaded video could not be opened. It may be corrupted.")

                st.info("Processing video — this may take a moment depending on length and hardware.")
                frame_placeholder = st.empty()
                alert_placeholder = st.empty()
                stop_button = st.button("Stop", key="stop_video")

                frame_idx = 0
                any_alert = False
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok or stop_button:
                        break
                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue

                    result = run_inference(model, frame, confidence_threshold)
                    any_alert = any_alert or result.is_alert

                    frame_placeholder.image(
                        cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    if result.is_alert:
                        alert_placeholder.error(build_alert_message(result))
                    else:
                        alert_placeholder.info(build_alert_message(result))

                cap.release()
                st.success("Video processing complete." if not any_alert else "Video processing complete — fire/smoke was detected in at least one frame.")

            except InvalidInputError as exc:
                st.error(f"❌ {exc}")
            except DetectionError as exc:
                st.error(f"❌ Detection failed: {exc}")

    # ================================ WEBCAM TAB (FR-03) ================================ #
    with tab_webcam:
        st.subheader("Live Webcam Detection")
        st.caption("Near-real-time detection from your default webcam.")

        run_webcam = st.checkbox("Start webcam", key="webcam_toggle")
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()

        if run_webcam:
            try:
                cap = open_webcam(0)
                st.info("Webcam running. Uncheck 'Start webcam' to stop.")

                while st.session_state.get("webcam_toggle", False):
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("Could not read a frame from the webcam.")
                        break

                    result = run_inference(model, frame, confidence_threshold)
                    frame_placeholder.image(
                        cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    if result.is_alert:
                        alert_placeholder.error(build_alert_message(result))
                    else:
                        alert_placeholder.success(build_alert_message(result))

                    time.sleep(0.03)

                cap.release()

            except WebcamNotAvailableError as exc:
                st.error(f"❌ {exc}")
            except DetectionError as exc:
                st.error(f"❌ Detection failed: {exc}")


if __name__ == "__main__":
    main()
