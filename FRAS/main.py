import cv2
import os
import csv

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

# Create a directory to store captured images
output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with rectangles around the detected faces
    cv2.imshow('Face Recognition', frame)

    # Check for user input to capture the image
    key = cv2.waitKey(1)
    if key == ord('c'):  # Press 'c' key to capture the image
        # Prompt the user to enter ID and Name for the detected face
        person_id = input("Enter ID for this person: ")
        person_name = input("Enter Name for this person: ")

        # Save the detected face as an image with ID and Name
        img_path = os.path.join(output_dir, f'face_{person_id}_{person_name}.png')
        cv2.imwrite(img_path, frame[y:y + h, x:x + w])

        # Append the ID, Name, and Image Path to the CSV file
        with open('captured_images/captured_info.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([person_id, person_name, img_path])

    # Break the loop if 'q' key is pressed
    elif key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()