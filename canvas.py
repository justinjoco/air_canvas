'''
ECE 5725 FINAL PROJECT
Stephanie Lin (scl97), Justin Joco (jaj263)
AIR CANVAS

canvas.py
'''

from multiprocessing import Process, Queue, Value, Lock, Array
import cv2
import numpy as np
import time
from datetime import datetime

# Setup OpenCV threshold values
threshold = 60
blurValue = 41
bgSubThreshold = 50
learningRate = 0

'''
# Master Process
def grab_display(run_flag, send_frame_queue, receive_contour_queue, p_start_turn):
    
    last_contour_receive_time = 0 
    startTime_ms   = 0
    contourRead    = False
    start_datetime = datetime.now()

    while(run_flag.value):
        # 1. Extract frame from video
        returnBool, frame = videoCap.read()
        # 2. Check if time since last send to queue exceeds 30ms
        current_time = datetime.now()
        time_dif     = current_time - start_datetime
        time_dif_ms  = time_dif.total_seconds()*1000
        # 3. Pass frame to queue
        # Only place frame in queue if time has passed 30ms and there are less than 4 frames in queue
        if ((time_dif_ms > 30) and (send_frame_queue.qsize() < 4)):
            start_datetime = current_time   # update the last send-to-queue time
            send_frame_queue.put(frame)     # put frame in queue
        # Check if receive_contour_queue is not empty
        if ((not receive_contour_queue.empty())):
            last_contour_receive_time = time.time()     # update last-contour-received time
            contours = receive_contour_queue.get()      # extract contour
        
        
    
# Worker Process 1
def frame_process_1(run_flag, send_frame_queue, receive_contour_queue, p_start_turn):
   # while(run_flag.value):
    return

# Worker Process 2 
def frame_process_2(run_flag, send_frame_queue, receive_contour_queue, p_start_turn):
   # while(run_flag.value):
    return

# Worker Process 3 
def frame_process_3(run_flag, send_frame_queue, receive_contour_queue, p_start_turn):
  #  while(run_flag.value):
    return


# MAIN: 1. Set video resolution
x_res = 320
y_res = 240
center_x = x_res/2
center_y = y_res/2

# MAIN: 2. Video capture
videoCap = cv2.VideoCapture(0)
'''



cv2.namedWindow("frame")

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

bgModel = cv2.BackgroundSubtractorMOG()

ret, first = videoCap.read()
first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray,(21,21), 0)

start = time.time()
counter = 0

while(True):
    try:
       
        if (counter >=60):
            end = time.time()
            seconds = end - start
            print(60/seconds)
            start = time.time()
            counter = 0
        #Get the frame
        ret, frame = videoCap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)

        difference = cv2.absdiff(gray,first_gray)

        thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
     #   thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       

        mask_time1 = time.time()

        #Create a foreground mask
     #   if start:

   #     fgMask = bgModel.apply(frame)
      #      start = False
        
        print("fgMask time: " + str(time.time()-mask_time1))

        '''
        mask_time2 = time.time()

        #  kernel = np.ones((3,3), np.uint8)
      #  fgMask = cv2.erode(fgMask, kernel, iterations=1)

        #Apply mask to frame to get hand
     #   fg_img = cv2.bitwise_and(frame, frame, mask=fgMask)

    #    print("fg_img time: " + str(time.time()-mask_time2))


     #   mask_time3 = time.time()

        #Convert to grayscale
      #  gray_img = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        
        print("gray_img time: " + str(time.time()-mask_time3))

        
      #  mask_time4 = time.time()

        #Smooth gray image with gaussian filter
     #   smooth_img = cv2.GaussianBlur(gray_img, (blurValue, blurValue),0)
        
    #    print("smooth_img time: " + str(time.time()-mask_time4))

     #   mask_time5 = time.time()

        #Create binary image
        ret, binary = cv2.threshold(smooth_img, threshold, 255, cv2.THRESH_BINARY)
        
        print("ret,binary time: " + str(time.time()-mask_time5))


       # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        counter+=1

       
        '''
       

        cv2.imshow("frame", thresh)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    except KeyboardInterrupt:
        break

videoCap.release()
cv2.destroyAllWindows()
    
