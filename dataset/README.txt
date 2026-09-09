DATASET FOLDER
==============

This folder holds the images and YOLO-format annotation files used to
train and evaluate the fire/smoke detection model.

Structure
---------
dataset/
├── data.yaml            <- YOLO dataset config (classes + folder paths)
├── images/
│   ├── train/            <- 70% of images
│   ├── val/               <- 20% of images
│   └── test/               <- 10% of images
└── labels/
    ├── train/
    ├── val/
    └── test/

Each image (e.g. images/train/fire_001.jpg) must have a matching label
file with the same base name (labels/train/fire_001.txt).

Label format (YOLO)
--------------------
One line per object in the image:

    <class_id> <x_center> <y_center> <width> <height>

- class_id: 0 = fire, 1 = smoke (must match dataset/data.yaml)
- All coordinates are normalized (0-1), relative to image width/height.

Example (fire_001.txt):
    0 0.512 0.430 0.220 0.300

Where to get data
------------------
Public datasets can be sourced from Kaggle, GitHub, or Hugging Face
(check each dataset's license before use). Search terms such as
"fire smoke detection dataset yolo" or "fire and smoke object detection
dataset" are a good starting point. You can also collect your own images
and annotate them with a free tool such as CVAT, LabelImg, or Roboflow.

Recommended composition
------------------------
Include a mix of:
  - Fire images (indoor/outdoor, varied sizes and distances)
  - Smoke images (varied density, lighting, backgrounds)
  - Normal/background images with NO fire or smoke (helps reduce false
    positives - these images can have an empty .txt label file, or no
    label file at all)

Once your images and labels are placed in the correct folders, run:
    python train.py
