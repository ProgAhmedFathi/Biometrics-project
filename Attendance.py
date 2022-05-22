import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# from PIL import ImageGrab


# Importing images
path = 'Images attendance'  # Import the images form the path
images = []  # Set the imported images in a list
images_names = []  # Set the name of the images in list
my_list = os.listdir(path)
print(my_list)

# Get just the name of the image without the extensions(.jpg.. etc)
for image in my_list:
    current_image = cv2.imread(f'{path}/{image}')
    images.append(current_image)
    images_names.append(os.path.splitext(image)[0])
print(images_names)


# Compute encodings
def find_encodings(images):
    encode_list = []

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)
        return encode_list


# Marking attendance
def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        my_data_list = f.readlines() # We read the data because if someone is already arrived we don't need to repeat it
        # print(my_data_list)

        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_string}')

# For capturing screen rather than webcam
# def capture_screen(bbox=(300,300,690+300,530+300)):
#     capture_screen = np.array(ImageGrab.grab(bbox))
#     capture_screen = cv2.cvtColor(capture_screen, cv2.COLOR_RGB2BGR)
#     return capture_screen


encode_list_known = find_encodings(images)
print('Encoding Complete')

# Create a video capture object so that we can grab frames from the webcam.
capture_image = cv2.VideoCapture(0)

# The while loop is to run the webcam.
while True:
    # Webcam image
    success, image = capture_image.read()
    # img = captureScreen()
    resized_image = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Webcam encodings
    capture_image_faces = face_recognition.face_locations(resized_image)
    capture_image_encode = face_recognition.face_encodings(resized_image, capture_image_faces)

    # Find matches
    for face_encode, face_location in zip(capture_image_encode, capture_image_faces):
        matches = face_recognition.compare_faces(encode_list_known, face_encode)
        face_distance = face_recognition.face_distance(encode_list_known, face_encode) # Lowest distance mean best match
        # print(face_distance)
        match_index = np.argmin(face_distance)  # Return the index of the lowest distance from all distances,
        #                                        which is the best match.

        if matches[match_index]:
            name = images_names[match_index].title()
            # print(name)

            # Make rectangle on the face based on the location
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Multiply the location of face because we resize the image above.
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

            # Put the name off the person with the rectangle on the image
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            #
            mark_attendance(name)

        # The next part of code use when Unknown person want to check attendance, replace it with the previous part.
        # Check if the distance to our min face is less than 0.5 or not,
        # If It's not then this means the person is unknown, so we change the name to unknown,
        # and don’t mark the attendance.
        # if face_distance[match_index] < 0.50:
        #     name = images_names[match_index].upper()
        #     mark_attendance(name)
        # else:
        #     name = 'Unknown'
        #     # print(name)
        #     y1, x2, y2, x1 = face_location
        #     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        #     cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', image)
    cv2.waitKey(1)
