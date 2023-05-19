import cv2
import os

import pickle
import face_recognition

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


from dotenv import load_dotenv
load_dotenv()

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('DB_URL'),
    'storageBucket':os.getenv('BUCKET_URL')
})


counter=0

#### A function which trains the model on all the faces available in faces folder

def findEncodings(path):
    encodeList = []
    bucket = storage.bucket()

    for subdir in os.listdir(path):
        images_path = os.path.join(path, subdir)
        for image_name in os.listdir(images_path):
            image = cv2.imread(os.path.join(images_path, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:
                encodeList.append(face_encoding[0])
            # else:
            #     print(f"No face found in {image_name}")
    return encodeList


def train_model():
    path = 'static/faces'
    images = []
    classNames = []

    myList = os.listdir(path)  # to extract list of names of all images
    print("Users found")
    print(myList)

    for cl in myList:
        images_path = os.path.join(path, cl)
        for image_name in os.listdir(images_path):
            image = cv2.imread(os.path.join(images_path, image_name))
            images.append(image)
            classNames.append(os.path.splitext(cl)[0])  # stores image names w/o labels

    encodeListKnown = findEncodings(path)
    print('Encoding Completed')

    encodeListKnownWithIds = [encodeListKnown, classNames]

    print("Saving file ..... ")
    file = open("static/EncodeFile.pkl", 'wb')
    pickle.dump(encodeListKnownWithIds, file)
    file.close()
    print("File Saved")
    print("Model trained")