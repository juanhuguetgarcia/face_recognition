import face_recognition
import cv2
import glob
import os
import logging
import numpy as np

images_path = 'images/'
encodings_path = 'encodings/'

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

def load_save_encoding(img_path):
    img_name = img_path[7:].replace('.jpg', '')
    encodings_file = encodings_path + img_name + '.npy'
    if os.path.exists(encodings_file):
        logging.info('Loading encoding...')
        encoding = np.load(encodings_file)
    else:
        logging.info('Generating encoding')
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)[0]
        logging.info('Saving for further use...')
        np.save(encodings_file, encoding)
    return img_name, encoding

logging.info('Loading images of known people...')
known_face_encodings = []
known_face_names = []
images = [img for img in glob.glob(images_path + '*') if img.endswith('.jpg')]

for ix, img_path in enumerate(images):
    logging.info('Image {i} of {n}'.format(i=ix+1, n=len(images)))
    img_name, encoding = load_save_encoding(img_path)
    known_face_encodings.append(encoding)
    known_face_names.append(img_name)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Get a reference to webcam #0 (the default one)
logging.info('Starting the webcam')
video_capture = cv2.VideoCapture(0)
logging.info("Starting face recognition")
i = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "mmm, nos conocemos?"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                if name.lower() == 'maria':
                    if i in range(0, 10):
                        name = 'hmmmm....'
                    if i in range(10, 30):
                        name = 'me suena tu cara...'
                    if i in range(30, 40):
                        name = 'tu debes ser...Maria!'
                    if i in range(40, 50):
                        name = """oye, menos zanganear..."""
                    if i in range(50, 65):
                        name = 'y a trabajar!'
                    if i > 65:
                        name = np.random.choice(['Maria', 'Maria', 'Maria',
                                                 'Reyes', 'de la O'])


            face_names.append(name)

    process_this_frame = not process_this_frame
    i += 1


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (14, 77, 146), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (14, 77, 146), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
