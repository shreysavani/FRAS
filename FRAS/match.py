import cv2
import dlib
import numpy as np

import csv
import os
from datetime import datetime
# Load the pre-trained face detection model
face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained face recognition model
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the CSV file with captured information
captured_info_file = 'captured_images/captured_info.csv'
known_faces_data = []

with open(captured_info_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        person_id, person_name, img_path = row
        known_face_image = cv2.imread(img_path)

        # Convert the image to RGB format (dlib requires RGB images)
        known_face_image_rgb = cv2.cvtColor(known_face_image, cv2.COLOR_BGR2RGB)

        # Detect the face and compute facial landmarks
        face_locations = face_detector(known_face_image_rgb)
        face_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")(known_face_image_rgb, face_locations[0])

        # Compute the face encoding from the face landmarks
        known_face_encoding = face_recognizer.compute_face_descriptor(known_face_image_rgb, face_landmarks)
        known_faces_data.append((person_id, person_name, known_face_encoding))

known_faces_names = [data[1] for data in known_faces_data]

# Open the camera for face recognition
video_capture = cv2.VideoCapture(0)

# Initialize variables for face recognition and attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = current_date + '.csv'
f = open(csv_file, 'a', newline="")
Inwriter = csv.writer(f)
# Check if the "recognized_student.csv" file exists
file_exists = os.path.isfile('recognized_student.csv')

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_detector(rgb_small_frame)
    face_encodings = [face_recognizer.compute_face_descriptor(rgb_small_frame, face_landmarks) for face_landmarks in
                      [dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")(rgb_small_frame, face_location)
                       for face_location in face_locations]]
    face_names = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face encoding with the known face encodings
        face_distances = [np.linalg.norm(np.array(data[2]) - np.array(face_encoding)) for data in known_faces_data]
        min_distance_index = np.argmin(face_distances)

        if face_distances[min_distance_index] < 0.6:  # You can adjust this threshold for better matching
            id_number = known_faces_data[min_distance_index][0]
            name = known_faces_data[min_distance_index][1]
        else:
            id_number = ""
            name = ""

        face_names.append(name)

        # Draw rectangle around the recognized face and display the name and ID
        top = face_location.top() * 4
        right = face_location.right() * 4
        bottom = face_location.bottom() * 4
        left = face_location.left() * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} (ID: {id_number})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Mark attendance if the recognized face matches with a known face
        if name in known_faces_names:
            known_faces_names.remove(name)
            today = datetime.today()  # Format datetime as string .strftime("%Y-%m-%d %H:%M:%S")
            Inwriter.writerow([id_number, name, today])

            # Exit the code and print the student data into a separate CSV file
            if not file_exists:  # Write the header row only if the file does not exist
                with open('recognized_student.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['ID Number', 'Student Name',  'Date & Time'])
                    file_exists = True  # Set flag to True after writing the header row

            with open('recognized_student.csv', mode='a', newline='') as file:  # Use 'a' for append mode
                writer = csv.writer(file)
                writer.writerow([id_number, name, today])

            break

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
video_capture.release()
cv2.destroyAllWindows()
f.close()

# Printing CSV file path as we cannot use os module to retrieve the current working directory
print("Attendance data saved in:", csv_file)
