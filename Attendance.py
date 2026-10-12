
import cv2
import numpy as np
import pickle
import os
import csv
from datetime import datetime, date

# ---------- Load face detector ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- Load model ----------
with open("face_recognition_model.pkl", "rb") as f:
    saved = pickle.load(f)

if len(saved) == 3:
    knn, names, roll_numbers = saved
else:
    knn, names = saved
    roll_numbers = {}

# ---------- Attendance setup ----------
attendance_file = "attendance.csv"
logged_names = set()

# Create CSV with headers if missing
if not os.path.isfile(attendance_file):
    with open(attendance_file, mode='w', newline='') as file:
        csv.writer(file).writerow(["Name", "Roll No", "Time", "Date"])


def load_today_logged_names():
    """Return set of names already marked today (from CSV)."""
    today = date.today().strftime("%Y-%m-%d")
    already = set()
    if os.path.isfile(attendance_file):
        with open(attendance_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None)                       # skip header
            for row in reader:
                if len(row) >= 4 and row[3] == today:
                    already.add(row[0])
    return already


# On startup, preload today's names so restarts don't duplicate
logged_names = load_today_logged_names()
current_day = date.today()
print(f"Loaded {len(logged_names)} entries already marked today: {logged_names}")

# ---------- Webcam loop ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- Midnight reset ----------
    today = date.today()
    if today != current_day:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"New day ({today}). Resetting attendance log.")
        logged_names.clear()
        current_day = today

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face.flatten().reshape(1, -1)

        label = knn.predict(face)[0]
        name = names[label]
        roll_no = roll_numbers.get(name, "N/A")

        # Mark only once per day
        if name not in logged_names:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%Y-%m-%d")

            with open(attendance_file, mode='a', newline='') as file:
                csv.writer(file).writerow(
                    [name, roll_no, current_time, current_date])

            logged_names.add(name)
            print(f"Attendance recorded for {name} "
                  f"(Roll No: {roll_no}) at {current_time}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({roll_no})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()