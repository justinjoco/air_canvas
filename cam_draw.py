'''
ECE 5725 FINAL PROJECT
Stephanie Lin (scl97), Justin Joco (jaj263)
AIR CANVAS

sample_linked.py
'''

from multiprocessing import Process, Queue, Value, Lock, Array
import cv2
import numpy as np
import time
from datetime import datetime
import sys
import math
import pygame 
import RPi.GPIO as GPIO
import os

# Set environment variables
os.putenv('SDL_VIDEODRIVER','fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')

# Initialize game
pygame.init()

# Screen settings 
size = width, height = 320, 240
black = 0,0,0
screen = pygame.display.set_mode(size)

# Brush settings 
radius = 2
coords = [(200,120),(200,121),(200,122),(200,123)]
RED = 255,0,0

# Margin settings
L_MARGIN = 10
R_MARGIN = 310
T_MARGIN = 10
B_MARGIN = 230

screen.fill(black)

# ======= SAMPLE_CAM3.PY CODE ============ #

# Setup screen variables 
hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
  #  print("width: " + str(width) "\n height" 
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)
		
        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

# Draw circles on screen
def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)
	
	
# ================= TRACE HAND ================= #

def get_centroid(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

	# obtain centroid
    ctr = centroid(max_cont)
    return ctr, max_cont
    

def manage_image_opr(frame, hand_hist):
    '''hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

	# obtain centroid
    cnt_centroid = centroid(max_cont)'''
    
    cnt_centroid, max_cont = get_centroid(frame, hand_hist)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)
	
    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        
       
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)
        return far_point
    else:
	return None
# =================== PYGAME DRAWING ==================== #

def in_bounds(coord):
    return (coord[0] >= L_MARGIN) and (coord[0] <= R_MARGIN) and (coord[1] >= T_MARGIN) and (coord[1] <= B_MARGIN)
    

# Draw a dot
# Screen res is 640x480

def l2_distance(prev_coord, curr_coord):
    return math.sqrt((curr_coord[0]-prev_coord[0])**2 + (curr_coord[1]-prev_coord[1])**2)


def draw_dot(coord): 
    if coord != None:
	coord_new = int(coord[0]/2), int(coord[1]/2)
	print("Dot drawn at: " + str(coord_new) ) 
	#time.sleep(.02)
    
	if in_bounds(coord_new):
	    pygame.draw.circle(screen, RED, coord_new, radius)
	    pygame.display.flip()
		      
	
	

# ================== MAIN ================== #
def main():
    global hand_hist
    draw = False
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)
    
    screen.fill(black)
    videoWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    videoHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    print("default resolution " + str(int(videoWidth)) + " x "+ str(int(videoHeight)))
	
    prev = None
    curr = None
    draw_thresh = 10
    while capture.isOpened():
	try:
			
			# wait for keypress 
	    pressed_key = cv2.waitKey(1)
	    _, frame = capture.read()

	    # Press z to create hand histogram
	    if pressed_key & 0xFF == ord('z'):
		is_hand_hist_created = True
		hand_hist = hand_histogram(frame)

	    if is_hand_hist_created:
		far_point = manage_image_opr(frame, hand_hist)
		
		# Draw dot located at centroid 
		ctr, mc = get_centroid(frame, hand_hist)
		
		if pressed_key & 0xFF == ord("d"): draw = not draw
		if far_point is not None:
		    curr = far_point
		    if draw and l2_distance(prev, curr) <= draw_thresh: 
			draw_dot(far_point)
		
		    if prev is None:
			prev = far_point
		    else:
			prev = curr
		# Draw dot located at farthest point (would be better)

	    else:
		frame = draw_rect(frame)

	    cv2.imshow("Live Feed", rescale_frame(frame))
	    

	    if pressed_key == 27:
		break
				
	except KeyboardInterrupt:
	    break

    cv2.destroyAllWindows()
    capture.release()
    


# Run main() 
if __name__ == '__main__':
    main()




