from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import cv2
import math
import argparse
import pyaudio
import pyttsx3

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()



# In train.py
model = load_model('gender_detection.keras')  # or .h5
ageProto = "./age_deploy.prototxt"
ageModel = "./age_net.caffemodel"
ageNet=cv2.dnn.readNet(ageModel,ageProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

parser=argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()

# open webcam
webcam=cv2.VideoCapture(args.image if args.image else 0)
classes = ['man','woman']

padding=20
count=0;
# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    if not status:
        cv2.waitKey()
        break
    # apply face detection
    face, confidence = cv.detect_face(frame)
    # frame,bboxs=faceBox(faceNet,frame)
    a=1;
    if not face:
        print("No face detected")

  
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])
      
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        
        blob=cv2.dnn.blobFromImage(face_crop, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        if(count==0):
            if(label=="man"):
                engine.setProperty('voice', voices[0].id)
                speak("You Are Man");
                speak("How Are You")
                speak("Where Are You From??")
            else:
                engine.setProperty('voice', voices[1].id)
                speak("You Are Woman")
                speak("How Are You")
                speak("Where Are You From??")

            count=count+1

        label = "{},{}: {:.2f}%".format(label,age, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()