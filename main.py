import numpy as np
import dlib    #To detect and localize facial landmarks
import cv2
import threading
from threading import Thread
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import pygame

'''
Here we have imported scipy package to compute the Euclidean distance between facial landmarks point in the eye aspect ratio calculation.
we have imported Thread class so we can play our alarm in a separate thread from the main thread to ensure our script doesn't pause execution while the alarm sounds.
Pygame to play the alarm.
dlib library to localize the facial landmarks.
'''

def sound_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("path")
    pygame.mixer.music.play()
'''
We have defined the sound_alarm function in which we are initializing  module of mixer (pygame.mixer.init()) .
pygame.mixer.music.load("path") will load the music file for playback
pygame.mixer.music.play() start the playback.
'''
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear
'''
We have defined the eye_aspect_ratio function in which we are computing the Euclidean distance between the two sets of vertical eye landmarks (x, y)-coordinates.
We have imported dist package to compute the Euclidean distance between the two sets of vertical eye landmarks (x, y)-coordinates.
We have imported dist package to compute the Euclidean distance between the horizontal eye landmark (x, y)-coordinates.
We have imported dist package to compute the Euclidean distance between the horizontal eye landmark (x, y)-coordinates.
'''



EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 40

COUNTER = 0
ALARM_ON = False

predictor_path = 'shape_predictor_68_face_landmarks (1).dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#Grab the indexes of the facial landmarks for the left and
# right eye,respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
( rStart , rEnd ) =face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index incv2.VideoCapture(0) \n')
        break
        # cv2.imshow(frame)

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)    # determine the facial landmarks for  #face region
        shape = face_utils.shape_to_np(shape) #converting to numpy array

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    d=threading.Thread(target=sound_alarm)
                    d.setDaemon(True)
                    d.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()



