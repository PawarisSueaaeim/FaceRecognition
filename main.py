import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import speech_recognition as sr
import pyttsx3

def makeFile():
    curr = datetime.now()
    dateStr = curr.strftime('%d-%m-%Y')
    w = open(f"report{dateStr}.txt", "w")
    w.write("name,date,time")
    w.close()

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    javis = pyttsx3.init()
    javis.say(f'Hello {name}')
    javis.runAndWait()
    curr = datetime.now()
    dateStr = curr.strftime('%d-%m-%Y')
    with open(f"report{dateStr}.txt","r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%d/%m/%Y')
            timeString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString},{timeString}')

path = 'imageAttandence'
images = []
className = []
myList = os.listdir(path)
print(myList)
r = sr.Recognizer()

for cls in myList:
    curlmg = cv2.imread(f'{path}/{cls}')
    images.append(curlmg)
    className.append(os.path.splitext(cls)[0])
print(className)
encodeListknown = findEncoding(images)
print('encoding complete')

cap = cv2.VideoCapture(0)
makeFile()

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        # print(faceLoc)
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        # print(index)
        # print(faceDis[index])

        if faceDis[matchIndex] <= 0.45:
            name = className[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('camera',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
