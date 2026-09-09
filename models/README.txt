MODELS FOLDER
=============

This folder holds the trained YOLO11n weights used by the application.

Expected file:
    models/best.pt

How to generate it:
    1. Prepare the dataset under dataset/images/ and dataset/labels/
       (see dataset/README.txt).
    2. Run:
           python train.py
       This trains YOLO11n and automatically copies the best checkpoint
       from runs/fire_smoke/weights/best.pt into this folder.

If you already have a trained best.pt (e.g. from Google Colab), simply
copy it here manually:
    runs/fire_smoke/weights/best.pt  ->  models/best.pt

The application (app.py), predict.py and evaluate.py all read the model
from this folder by default. best.pt is NOT included in version control
(see .gitignore) because trained weights are a large binary file that
should be generated locally or downloaded separately.
