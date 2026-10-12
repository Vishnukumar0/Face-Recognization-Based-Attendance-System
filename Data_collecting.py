# #here we used 3 liberary for taining the data
# import cv2
# import numpy as np
# import os

# # Load OpenCV's built-in face detector
# #here we import cascadeclassifier with the help of cv2
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# #make datdset directory
# dataset_path = "face_dataset"
# if not os.path.exists(dataset_path):
#     os.makedirs(dataset_path)    #create the dataset directory if it doesnot exist

# name = input("Enter your name: ")
# roll_no=int(input("Enter Roll No: "))
# person_path = os.path.join(dataset_path, name, str(roll_no))
# os.makedirs(person_path, exist_ok=True)   #create a directory for the person if it doesnot exist

# cap = cv2.VideoCapture(0)     #open the webcam 
# count = 0      #intialized the image count

# while count < 100:  # Collect 100 face images
#     ret, frame = cap.read()    #read a frame from webcam
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #convert the frame to gray scale
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)  #detect face from the frame

#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]  #extract the face region from the gray scale frame
#         face = cv2.resize(face, (100, 100))   #resize the face to 100 x 100 pixel
#         cv2.imwrite(f"{person_path}/{count}.jpg", face)  #save the face image to the person directory
#         count += 1   #increament the image count
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  #draw reactangle arround the face

#     cv2.imshow("Face Collection", frame)  #dispaly the frame with the detected face
#     if cv2.waitKey(1) & 0xFF == ord("q"):  #exit the loop if Q is pressed
#         break

# cap.release()   #release the webcam
# cv2.destroyAllWindows() #close all open-cv windows
# Here we use libraries for collecting face data

import cv2
import os

# Load OpenCV's built-in face detector
# Here we import CascadeClassifier with the help of cv2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check whether Haar Cascade loaded successfully
if face_cascade.empty():
    print("Error: Haar Cascade file load nahi hui.")
    exit()

# Make dataset directory
dataset_path = "face_dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Take user input
name = input("Enter your name: ")
roll_no = input("Enter Roll No: ")

# Create person's directory
person_path = os.path.join(dataset_path, name, roll_no)

os.makedirs(person_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

# Check whether webcam opened successfully
if not cap.isOpened():
    print("Error: Webcam open nahi ho raha.")
    exit()

count = 0

# Collect 100 face images
while count < 100:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Frame read nahi ho raha.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process detected faces
    for (x, y, w, h) in faces:

        # Extract face
        face = gray[y:y+h, x:x+w]

        # Resize face to 100 x 100
        face = cv2.resize(face, (100, 100))

        # Save face image
        cv2.imwrite(
            f"{person_path}/{count}.jpg",
            face
        )

        count += 1

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        
    # Display webcam
    cv2.imshow("Face Collection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()

print(f"{count} face images collected successfully.")