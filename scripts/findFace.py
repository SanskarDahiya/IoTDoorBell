import cv2
import os
import numpy as np
import datetime
cascPath = "scripts/haarcascade_frontalface_default.xml"


subject_names = ["", "Person1", "Person2"]


def detect_face_from_image(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # here we load Face classifier defined in open_Cv libraries, p.s. i use lbpcascade which is faster than others
    face_cascade = cv2.CascadeClassifier(cascPath)
    # Here i detect all faces from a image
    all_faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5)

    # if there is no face in image it returns None
    if (len(all_faces) == 0):
        return None, [0, 0, 0, 0]

    # we use only first face assuming that image has only one face
    (x, y, w, h) = all_faces[0]

    # return the face of image
    return gray[y:y + w, x:x + h], all_faces[0]


def prepare_training_data(data_folder_path):
    # gets all the directories in training_data folder
    directories = os.listdir(data_folder_path)
    # Defined two lists ones for faces and other labels corresponding to each person
    faces = []
    labels = []
    for directory_name in directories:
        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if directory_name.startswith("person"):
            print('I am working')
            # to get the label corresponding to each image we perform: Replace "s1" with "1"
            label = int(directory_name.replace("person", ""))

            # to build the path of directory containing images like "training-data/s1"
            subject_directory_path = data_folder_path + "/" + directory_name
            # Get images names using os.listdir
            subject_images_names = os.listdir(subject_directory_path)

            for image_name in subject_images_names:
                # to avoid unwanted files
                if not image_name.startswith("."):
                    photo_path = subject_directory_path + "/" + image_name
                    # reading image through cv2
                    image = cv2.imread(photo_path)

                    # display an image window to show the image
                    # cv2.imshow("Training on image...", image)
                    cv2.waitKey(100)
                    # detect face
                    face, rect = detect_face_from_image(image)

                    print('....')
                    if face is not None:
                        faces.append(face)
                        labels.append(label)

            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    return faces, labels


def find_Face_Name(img):
    # initiate id counter
    id = 0
    canSave = False
    # image in laptops is flipped automatically
    img = cv2.flip(img, 1, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_in_image = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
    )

    for(x, y, w, h) in faces_in_image:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
        print(confidence)
        canSave = True
        if (confidence < 50):
            id = subject_names[id]
            FaceAnswer['yes']['name'] = id
            FaceAnswer['yes']['len'] += 1
            confidence = " {0}%".format(round(100 - confidence))

        else:
            FaceAnswer['not'] += 1
            id = "I Don't Know You"
            confidence = " {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5),font, 1, (255, 255, 0), 1)
    cv2.imshow('camera', img)

    return canSave


print("Preparing data...")
faces, labels = prepare_training_data("scripts/faceData")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

face_cascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX

print("Preparing done...")
FaceAnswer = {
    "not": 0,
    "yes": {
        "name": "",
        "len": 0
    }
}


def Capture_Face():
    # Initialize real-time video capture
    filename = 'face\IMG-'+str(datetime.datetime.now().microsecond)+'.jpg'
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video-width
    cam.set(4, 480)  # set video-height
    exit = 0
    FaceAnswer['not'] = 0
    FaceAnswer['yes']['len'] = 0
    notSaved = True
    while True:
        exit += 1
        if(exit > 50):
            break
        ret, img = cam.read()
        canSave = find_Face_Name(img) or False
        if(canSave):
            notSaved = False
            cv2.imwrite(filename, img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:         # Press 'ESC' for exiting the program
            break
    if(notSaved):
        ret, img = cam.read()
        cv2.imwrite(filename, img)
    print("I am Cleaning up now")
    cam.release()
    cv2.destroyAllWindows()
    if(FaceAnswer['not'] > FaceAnswer['yes']['len']):
        return [filename, False]
    return [filename, FaceAnswer['yes']['name']]
