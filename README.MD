# Face Recognition Attendance System

A real-time face recognition attendance system using OpenCV, KNN, and a webcam.

## Features
- Detects and recognizes faces from a webcam feed
- Marks attendance automatically in `attendance.csv`
- Resets attendance daily at midnight
- Supports multiple people via folder-based dataset

## Project Structure
```
face-attendance/
├── train_model.py         # Train the KNN classifier
├── attendance.py          # Real-time attendance logger
├── face_dataset/          # Training images (name/roll_no/*.jpg)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone
git clone https://github.com/Vishnukumar0/Face-Recognization-Based-Attendance-System.git
cd face-attendance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your dataset
#    face_dataset/<name>/<roll_no>/*.jpg

# 4. Train
python train_model.py

# 5. Run
python attendance.py
```

Press `q` to quit the webcam window.

## How It Works
1. `train_model.py` reads images from `face_dataset/`, resizes to 100×100, flattens them, and trains a KNN classifier. It saves `(knn, names, roll_numbers)` into `face_recognition_model.pkl`.
2. `attendance.py` opens the webcam, detects faces with Haar Cascades, predicts the person with KNN, and appends a row to `attendance.csv` — once per person per day.

## Attendance CSV Format
| Name | Roll No | Time | Date |
|------|---------|------|------|
| alice | 101 | 09:14:22 | 2026-10-12 |

## License
MIT