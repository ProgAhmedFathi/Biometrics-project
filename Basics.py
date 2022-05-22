import cv2
import face_recognition

# Get the images and convert the color
img_basic = face_recognition.load_image_file('Images basics/Ahmed 1.jpg')
img_basic = cv2.cvtColor(img_basic, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file('Images basics/Ahmed 2.jpg')  # True test
# img_test = face_recognition.load_image_file('Images basics/Omar 1.jpg')  # False test
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# Find the face in the images and select it with rectangle
face_location = face_recognition.face_locations(img_basic)[0]
encode_face = face_recognition.face_encodings(img_basic)[0]
cv2.rectangle(img_basic, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (255, 0, 255), 2)

face_location_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (255, 0, 255), 2)

# Compare the results/encodes of the images and show this result on the image
results = face_recognition.compare_faces([encode_face], encode_test)
face_distance = face_recognition.face_distance([encode_face], encode_test)
print(results, face_distance)
cv2.putText(img_test, f'{results} {round(face_distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Display the images
cv2.imshow('Ahmed', img_basic)
cv2.imshow('Ahmed Test', img_test)
cv2.waitKey(0)
