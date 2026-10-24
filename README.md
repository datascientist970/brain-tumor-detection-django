# Brain Tumor Detection (Django + CNN)

A Django web app that detects brain tumors in uploaded MRI images with a **TensorFlow/Keras CNN**, and estimates the tumor area from the scan. Recovered from a RAR archive inside a projects collection into its own repository.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?logo=django&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)

## Overview

Upload a brain MRI image and the app classifies it as tumor / no-tumor. For positive cases it also segments the tumor region with OpenCV (thresholding + largest contour) and reports an estimated area in mm². Each upload and its result are stored (`UploadedImage` model).

## How It Works

1. **Training (at startup, in `detection/views.py`)** — images are loaded from two class folders (`yes` / `no`), preprocessed with OpenCV (resize 128×128, normalize), and a CNN (`Conv2D → MaxPooling → Flatten → Dense`) is trained with scikit-learn train/test split and label encoding
2. **Prediction** — an uploaded image goes through the same preprocessing and the trained model
3. **Area estimation** — binary thresholding finds the largest contour; pixel area is converted to mm²

## Key Features

- Image upload with result page (`upload.html` / `result.html`)
- CNN training + inference in one Django app
- Tumor area estimation in mm² via OpenCV contours
- Upload history stored with result and area

## Tech Stack

Python · Django · TensorFlow/Keras · OpenCV · NumPy · scikit-learn

## Project Structure

```
├── tumor_detection/  # Project settings, URLs, WSGI/ASGI
├── detection/        # App: views (train + predict + area), UploadedImage model, templates
└── manage.py
```

## Getting Started

```bash
git clone https://github.com/datascientist970/brain-tumor-detection-django.git
cd brain-tumor-detection-django
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Notes

- The training-image folders are referenced by **local paths** in `detection/views.py` (`train_folder_yes` / `train_folder_no`) — point them at your own MRI dataset (yes/no class folders) before running; the dataset is not included in this repository.
- Recovered from the `tumor_detection.rar` archive inside the `Web-Development-Django-` collection; the development database and sample uploads are not included.
- Learning/research prototype — **not a medical device**; not for diagnosis.

## Author

**Muhammad Dawood** — AI & Data Science Engineer | Full-Stack Developer
GitHub: https://github.com/datascientist970 · LinkedIn: https://www.linkedin.com/in/dawood406/
Mardan, Khyber Pakhtunkhwa, Pakistan
