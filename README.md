# 🔥 Fire and Smoke Detection Using Machine Learning

A computer-vision system that detects **fire** and **smoke** in images,
recorded videos, or live webcam feed using a lightweight **YOLO11n**
object-detection model, with a simple **Streamlit** interface for
non-technical users.

> TY (Third Year) College Mini Project — built entirely with free /
> open-source tools, suitable for a normal laptop or free Google Colab
> training.

---

## ✨ Features

- 🖼️ Image upload and detection (JPG / JPEG / PNG)
- 🎞️ Recorded video processing
- 📷 Live webcam detection
- 🎯 Bounding boxes with class label (Fire / Smoke) and confidence score
- 🎚️ Configurable confidence threshold
- 🚨 Automatic warning/alert when fire or smoke is detected
- 🔢 Support for multiple detections per image/frame
- ⚠️ Clear error handling for invalid files, missing model, or webcam issues
- 📊 Model evaluation with Precision, Recall, F1-score, mAP@50, mAP@50-95

---

## 🗂️ Project Structure

```
Fire_and_Smoke_Detection/
│
├── app.py                 # Streamlit UI (image / video / webcam)
├── train.py                # Trains the YOLO11n model
├── evaluate.py              # Evaluates model performance
├── predict.py                 # CLI prediction (image / video / webcam)
├── requirements.txt            # Python dependencies
├── README.md                    # This file
├── .gitignore
│
├── src/
│   └── detection_utils.py    # Shared backend logic (model load, inference,
│                              # preprocessing, alerts, error handling)
│
├── dataset/
│   ├── data.yaml              # YOLO dataset configuration
│   ├── README.txt              # Dataset preparation guide
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}
│
├── models/
│   ├── README.txt
│   └── best.pt                 # Trained model (created after training)
│
└── runs/
    └── fire_smoke/weights/     # Training checkpoints (best.pt, last.pt)
```

---

## ⚙️ Requirements

- Python 3.9+
- 8 GB RAM recommended
- Webcam (optional, for live detection)
- GPU optional — training also works on free Google Colab

See [`requirements.txt`](requirements.txt) for the full package list
(PyTorch, Ultralytics YOLO, OpenCV, Streamlit, NumPy, Pandas, Matplotlib).

---

## 🚀 Setup

```bash
# 1. Clone / unzip the project, then move into it
cd Fire_and_Smoke_Detection

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Prepare the Dataset

1. Collect fire, smoke, and normal/background images (Kaggle, GitHub,
   Hugging Face, or your own photos/frames — check licenses).
2. Annotate fire/smoke regions with bounding boxes in YOLO format
   (tools like CVAT, LabelImg, or Roboflow all export this format).
3. Place images and matching `.txt` label files into:
   ```
   dataset/images/train, dataset/images/val, dataset/images/test
   dataset/labels/train, dataset/labels/val, dataset/labels/test
   ```
   Recommended split: 70% train / 20% val / 10% test.

Full details: [`dataset/README.txt`](dataset/README.txt).

---

## 🏋️ Train the Model

```bash
python train.py --epochs 100 --imgsz 640 --batch 16
```

This fine-tunes YOLO11n on your dataset and automatically copies the
best checkpoint to `models/best.pt`. You can also train on Google Colab
and copy the resulting `best.pt` into the `models/` folder manually.

---

## 📊 Evaluate the Model

```bash
python evaluate.py --model models/best.pt --data dataset/data.yaml --split test
```

Prints Precision, Recall, F1-score, mAP@50, and mAP@50-95. Confusion
matrix and PR-curve plots are saved by Ultralytics under `runs/detect/val*/`.

---

## 🖥️ Run the Application

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`),
then choose the **Image**, **Video**, or **Webcam** tab, adjust the
confidence threshold in the sidebar, and start detecting.

---

## 🧪 Quick CLI Prediction

```bash
python predict.py --source path/to/image.jpg
python predict.py --source path/to/video.mp4 --save output.mp4
python predict.py --source webcam --show
```

---

## 🛠️ Tech Stack

| Layer            | Technology                          |
|------------------|--------------------------------------|
| Language          | Python                              |
| Detection Model   | YOLO11n (Ultralytics)                |
| Deep Learning     | PyTorch                              |
| Computer Vision   | OpenCV                               |
| Data Processing   | NumPy, Pandas                        |
| Frontend          | Streamlit                            |
| Training          | Google Colab (free tier) / local      |
| Version Control   | Git / GitHub                          |
| Database (optional) | SQLite                              |
| LLM (optional)    | Qwen2.5-1.5B via Ollama               |

---

## 🔮 Future Enhancements

- Real-time CCTV / multi-camera monitoring
- Mobile app with push notifications
- Email / SMS alerts
- Raspberry Pi + IoT (smoke/temperature) sensor integration
- Detection history dashboard with SQLite/cloud storage
- LLM-generated incident summaries (Qwen2.5-1.5B via Ollama)

---

## 📄 License

This is an academic mini project intended for learning purposes. Verify
the license of any third-party dataset before redistribution.
