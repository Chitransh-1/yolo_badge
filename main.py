import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
import datetime as datetime

video_capture = cv2.VideoCapture(0)

image_folder = "photos"

known_face_encodings = []
known_face_names = []

# Loop se sare images load karenge
for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):  # Agar PNG ya dusra format ho to "*.png" ya "*.jpeg" bhi include karo
    image_name = os.path.basename(image_path).split(".")[0]  # Filename ko naam ke liye use karenge
    image = face_recognition.load_image_file(image_path)
    
    encodings = face_recognition.face_encodings(image)
    
    if encodings:  # Kabhi kabhi image me face detect nahi hota, isliye check karna zaroori hai
        known_face_encodings.append(encodings[0])
        known_face_names.append(image_name)

print("Encodings loaded:", known_face_names)

students = known_face_names.copy()
print("Students Name : ",students)

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.datetime.now()
current_date = datetime.datetime.now().strftime("%d-%m-%Y")

f = open(current_date+'.csv', 'w+', newline='')
lnwrite = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)

            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    current_time = datetime.datetime.now().strftime("%H-%M-%S")
                    lnwrite.writerow([name, current_time, current_date])

    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
