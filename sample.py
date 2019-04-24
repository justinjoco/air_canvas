'''
sample.py
Gets live video capture

'''


import cv2
import numpy as np
import time

print("Running opencv version: " + str(cv2.__version__))

videoCap = cv2.VideoCapture(0)


if (not videoCap.isOpened()):
    print("Error: can't find camera")
    quit()
else:
    print("Success: pi camera is open")

videoWidth = videoCap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
videoHeight = videoCap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

print("default resolution " + str(int(videoWidth)) + " x "+ str(int(videoHeight)))

start = time.time()
counter = 0

while(True):
    try:
        if (counter >=120):
            end = time.time()
            seconds = end - start

            print(120/seconds)
            counter = 0
            start = time.time()

        ret, frame = videoCap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        counter+=1
    except KeyboardInterrupt:
        break

videoCap.release()
cv2.destroyAllWindows()
    
