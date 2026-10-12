import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = "face_dataset"
data = []
labels = []
roll_numbers = {}                   

# Every person is a subfolder of face_dataset
names = [name for name in os.listdir(dataset_path)
         if os.path.isdir(os.path.join(dataset_path, name))]
names.sort()                         # stable label order

for idx, name in enumerate(names):
    person_path = os.path.join(dataset_path, name)

    # Each roll_no is a subfolder inside the person folder
    for roll_no in os.listdir(person_path):
        roll_path = os.path.join(person_path, roll_no)
        if not os.path.isdir(roll_path):
            continue

        roll_numbers[name] = roll_no        # ✅ store mapping

        for file in os.listdir(roll_path):
            file_path = os.path.join(roll_path, file)
            if os.path.isfile(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    data.append(img.flatten())
                    labels.append(idx)
                else:
                    print(f"Failed to load image: {file}")

data = np.array(data)
labels = np.array(labels)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data, labels)

# ✅ Save 3-tuple: model + names + roll_numbers
with open("face_recognition_model.pkl", "wb") as f:
    pickle.dump((knn, names, roll_numbers), f)

print("Training complete and model saved!")