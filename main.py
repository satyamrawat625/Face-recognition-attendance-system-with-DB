from datetime import datetime

import cv2
import os
from flask import Flask, request, render_template, url_for, redirect
import numpy as np

import pickle
import face_recognition
import re
import trainModel
from utilityMethods import extract_attendance ,add_attendance ,identify_face ,totalreg ,datetoday2,datetoday

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

from dotenv import load_dotenv
load_dotenv()

cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('DB_URL')
    })

app1= firebase_admin.get_app()
bucket = storage.bucket(app=app1)

#### Defining Flask App
app = Flask(__name__)

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv', 'w') as f:
        f.write('ID,Name,Time')


#### extract the face from an image
def extract_faces(img): #chng
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)#haarcascade used to detect face
    return face_points



################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/',endpoint='home')
def home():
    ID,names,times, l = extract_attendance()
    return render_template('index.html', totalreg=totalreg(),
                           datetoday2=datetoday2())



#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():

    if 'EncodeFile.pkl' not in os.listdir('static'):
        return render_template('index.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Load the encoding file
    print("Loading Encoded File ...")
    file = open('static/EncodeFile.pkl', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, classNames = encodeListKnownWithIds
    # print(classNames)
    print("Encode File Loaded")

    while True:
        success, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)


        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            # matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)

            if len(faceDis) > 0:#faces exist

                matchIndex = np.argmin(faceDis)

                if faceDis[matchIndex] < 0.50:
                    uID = (re.split('_', classNames[matchIndex]))[0]  # stores uID
                    name = ((re.split('_', classNames[matchIndex]))[1]).upper()

                    add_attendance(classNames[matchIndex].upper())

                    # print(uID)
                    studentInfoFromDb = db.reference(f'Students/{uID}').get()
                    # print(studentInfoFromDb['attendance'])

                    # Update attendance
                    datetimeObject = datetime.strptime(studentInfoFromDb['last_attendance_time'],
                                                       "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                    # print(secondsElapsed)

                    if secondsElapsed > 72000:  # attendance has not been updated within the last 20 hours
                        ref = db.reference(f'Students/{uID}')
                        studentInfoFromDb['attendance'] += 1
                        ref.child('attendance').set(studentInfoFromDb['attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                else:
                    name = 'Unknown'
                    uID = 'NA'

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{name} id_{uID}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.64, (255, 255, 255),
                        2)  # to print on webcam window
                # cv2.putText(img, f'{name} ({1- faceDis[matchIndex]:.2f}) {uID}', (x1 + 6, y2 - 6),cv2.FONT_HERSHEY_COMPLEX, 0.64, (255, 255, 255), 2)  # to print on webcam window


        cv2.namedWindow('Attendance System',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attendance System', 900, 600)
        cv2.imshow('Attendance System', img)  # shows webcam
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    cv2.destroyAllWindows()


    return render_template('attendanceMarked.html')


#### This function will run when we add a new user, Post as we are submitting data to server
@app.route('/add', methods=['GET', 'POST'])
def add():
    ref = db.reference('Students')

    newuserid = request.form['newuserid']
    newusername = request.form['newusername']
    newuserdept = request.form['newuserdept']
    newuserdob = request.form['newuserdob']

    userimagefolder = 'static/faces/' + newuserid + '_' + str(newusername)

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    # Store the new student record in Firebase

    new_student = {
            'username': newusername,
            'department': newuserdept,
            'dob': newuserdob,
            'attendance': 0,
            'last_attendance_time': '2022-12-11 01:54:34'

        # 'imagefolder': userimagefolder
        }
    ref.child(newuserid).set(new_student)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/20', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)

            if j % 5 == 0:
                name = newuserid + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 100:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'): #to exit
            break
    cv2.destroyAllWindows()

    return render_template('userAdded.html', redirect_url= url_for('home'))


@app.route('/attendanceTod', methods=['GET'])
def showAttendance():
    ID, names, times, l = extract_attendance()
    return render_template('attendanceTod.html', ID=ID, names=names, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())

@app.route('/trained', methods=['GET', 'POST'])
def trainModelOnImg():
    print('Training Model')
    trainModel.train_model()
    return render_template('trained.html', redirect_url= url_for('home'))


#Manually update attendance
def search_user(student_id):
    ref = db.reference('Students')
    user_ref = ref.child(student_id)
    user_data = user_ref.get()
    return user_data

def update_attendance(student_id, new_attendance):
    ref = db.reference(f'Students/{student_id}')
    ref.child('attendance').set(new_attendance)
    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    student_name= ref.get().get('username')
    print(student_name)

    add_attendance(f'{student_id}_{student_name}')

# page to enter user id & attendance
@app.route('/changeDetails', methods=['GET', 'POST'])
def manuallyAddAttendance():
    student_id = request.form['student_id']
    new_attendance = int(request.form['new_attendance'])

    user_data = search_user(student_id)

    if user_data is not None:
        update_attendance(student_id, new_attendance)
        return redirect('attendanceTod')
    else:
        return render_template('user_not_found.html')

# manually update attendance of student by searching using student id
@app.route('/manualUpdate', methods=['GET', 'POST'])
def manualUpdate():
    return render_template('manualUpdate.html')

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)