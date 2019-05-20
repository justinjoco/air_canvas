'''
ECE 5725 FINAL PROJECT
Stephanie Lin (scl97), Justin Joco (jaj263)
AIR CANVAS

cam_draw.py
'''

import cv2
import numpy as np
import time
from datetime import datetime
import sys
import math
import pygame 
from pygame.locals import *
import RPi.GPIO as GPIO
import os

# Set environment variables
os.putenv('SDL_VIDEODRIVER','fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')
os.putenv('SDL_MOUSEDRV', 'TSLIB')      # track mouse clicks 
os.putenv('SDL_MOUSEDEV', '/dev/input/touchscreen')


#Set GPIO mode
GPIO.setmode(GPIO.BCM)


GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP) #Change color
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP) #Size up
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP) #Size down
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP) #Quit


# Initialize game
pygame.init()


# Screen settings 
size = width, height = 320, 240
black = 0,0,0
screen = pygame.display.set_mode(size)

pygame.mouse.set_visible(False)

# Brush settings 
radius = 2

#Colors
RED = 255,0,0
GREEN = 0,255,0
BLUE = 0,0,255
WHITE = 255,255,255
BLACK = 0,0,0

#Create color list
colors = [RED, GREEN, BLUE]

#Initialize drawing color
color_index = 0
curr_color = RED

# Margin settings
L_MARGIN = 10
R_MARGIN = 310
T_MARGIN = 10
B_MARGIN = 230



BTN_SIZE = 50
CENTER_POS = 160,120

#Fill first screen with black
screen.fill(black)


#Create pygame font
font = pygame.font.Font(None, 20)


# ======= SAMPLE_CAM3.PY CODE ============ # <--- NOT OURS
#Reference: https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m
hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None



#Rescales the output frame to 320 x 240 screen
def rescale_frame(frame, wpercent=130, hpercent=130):
	width = int(frame.shape[1] * wpercent / 100)
	#  print("width: " + str(width) "\n height" 
	height = int(frame.shape[0] * hpercent / 100)
	return cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

#Finds the contours of the hand
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

#Draws the rectangles for calibration
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

#Attains a histogram of the colors that encapsulate the above retangles
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

#Find the farthest point of hand from the centroid
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

# Draw circles on screen at specified point on the screen
def draw_circles(frame, traverse_point):
	if traverse_point is not None:
		for i in range(len(traverse_point)):
			cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)
	
	
# ================= TRACE HAND ================= # <-- NOT OURS
#Reference: https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m

#Finds the center of the hand
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

#Checks if a coordinate is within the margins we define
def in_bounds(coord):
	return (coord[0] >= L_MARGIN) and (coord[0] <= R_MARGIN) and (coord[1] >= T_MARGIN) and (coord[1] <= B_MARGIN)
    

# Draw a dot
# Screen res is 640x480

#Measures the Euclidean distance between two points 
def l2_distance(prev_coord, curr_coord):
	return math.sqrt((curr_coord[0]-prev_coord[0])**2 + (curr_coord[1]-prev_coord[1])**2)

#Draws a line between two drawn dots
def interpolate(prev_coord, curr_coord):
    
	if (prev_coord is not None) and (curr_coord is not None):
		prev_scaled = 320 - int(prev_coord[0]/2), int(prev_coord[1]/2)
		curr_scaled = 320 - int(curr_coord[0]/2), int(curr_coord[1]/2)
		
		pygame.draw.line(screen, curr_color, prev_scaled, curr_scaled, radius*2)
		pygame.display.flip()

#Draws a dot at a given point in the Pygame display
def draw_dot(coord): 
	if (coord != None):
		coord_scaled = 320 - int(coord[0]/2), int(coord[1]/2)
		#prev_scaled = 320 - int(prev_coord[0]/2), int(prev_coord[1]/2)
		print("Dot drawn at: " + str(coord_scaled) ) 
		#time.sleep(.02)
	if in_bounds(coord_scaled):
		pygame.draw.circle(screen, curr_color, coord_scaled, radius)
		#pygame.draw.line(screen, BLUE, prev_scaled, coord_scaled, radius*2)
		pygame.display.flip()
		      
#Changes the color by iterating through the color list defined earlier
def change_color():
	global curr_color, color_index
	color_index +=1 
	if color_index >= len(colors):
		color_index = 0
		curr_color = colors[color_index]
		print(curr_color)
 
#Increases or decreases the drawn dot and line sizes   
def change_radius(up_or_down):
	global radius
	if up_or_down: radius+=1
	else: radius-=1

	

# ================== MAIN ================== #
def main():
	global hand_hist

	#Do not draw on init
	draw = False
	is_hand_hist_created = False

	#Create a capture variable
	capture = cv2.VideoCapture(0)

	screen.fill(black)
	videoWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
	videoHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    

	
	#Intialize the current and previous drawn points
	prev = None
	curr = None
	prev_dot = None
	curr_dot = None
	draw_thresh = 10


	pygame.display.flip() 

		#Calibrate histogram on input
	calibrate = True

	while capture.isOpened():
		try:
				
				# wait for keypress 
			pressed_key = cv2.waitKey(1)

			#Read a frame from video capture
			_, frame = capture.read()




			if is_hand_hist_created:
				far_point = manage_image_opr(frame, hand_hist)
				
				# Draw dot located at farthest point
				ctr, mc = get_centroid(frame, hand_hist)
				
				
				if far_point is not None:
					curr = far_point
			
					#If we're drawing, make sure that we only draw dots if two subsequent dots are within a certain distance from each other
					#Interpolate between two drawn dots
					if draw:
						if l2_distance(prev, curr) <= draw_thresh:
							prev_dot = curr_dot
							curr_dot = far_point
							draw_dot(far_point)
							interpolate(prev_dot,curr_dot)
						else:
							interpolate(prev_dot, curr_dot)
			
				
					if prev is None:
						prev = far_point
					else:
						prev = curr
	

			else:
				frame = draw_rect(frame)

		    
		    
		    #Go through the pygame events
			for event in pygame.event.get():
				if (event.type is MOUSEBUTTONDOWN):
					pos = pygame.mouse.get_pos()
				elif(event.type is MOUSEBUTTONUP):
					pos = pygame.mouse.get_pos()
					x, y = pos

					#If we're calibrating, go to draw screen and create hand histogram if calibrate button is pressed
					if calibrate:
						if y >= 180 and y <=220 and x>=120 and x<=200:
							is_hand_hist_created = True
							hand_hist = hand_histogram(frame)
							calibrate = False
							screen.fill(black)
							pygame.display.flip() 

					#If we're drawing, 
					#if we hit the draw button, trigger drawing on and off
					#if we hit the calibrate button, disable drawing, reintialize dot variables, and go back to calibrate screen
					#If we hit anywhere on the screen that is not a button, rotate through the color list
					else:
						if y >= 120 and x <160:
							print("Draw/Not Draw")
							draw = not draw
					
						elif x >= 160 and y >120:
							print("Calibrate")
							draw = False
							is_hand_hist_created = False
							calibrate = True
							prev = None
							curr = None
							prev_dot = None
							curr_dot = None
						else:
							change_color()

			#Rescale the display frame to 320 x 240 pixels
			rescaled_frame = rescale_frame(frame)
			
			#Draw the calibrate button on the live cam screen if we're calibrating
			if calibrate:
				#print(rescaled_frame.shape)
				surface = pygame.surfarray.make_surface(rescaled_frame.transpose(1,0,2)[...,::-1])
				surface.convert()
				
				cal_surface = font.render('Calibrate', True, WHITE)
				
				rect_cal = cal_surface.get_rect(center=(160,200))
				
				screen.blit(surface, (0,0))
				pygame.draw.rect(screen, BLUE, pygame.Rect(120, 190, 80, 20))  
				screen.blit(cal_surface, rect_cal)
				
				
				pygame.display.flip() 
			
			#Render the draw and quit buttons on the drawing page
			else:
			
				pause_surface = font.render('Draw', True, WHITE)
				rect_pause = pause_surface.get_rect(center=(40,200))
				screen.blit(pause_surface, rect_pause)
				
				cal_surface = font.render('Calibrate', True, WHITE)
				rect_cal = cal_surface.get_rect(center=(260,200))
				screen.blit(cal_surface, rect_cal)
				pygame.display.flip()
			
		
		       

		  	#If we hit button 17, change the color
			if not GPIO.input(17):
				change_color()
		    
		    #If we hit button 22, increase the drawn dot size
			if not GPIO.input(22):
				change_radius(True)
		    
		    #If we hit button 23, decrease dot size
			if not GPIO.input(23):
				change_radius(False)
			
			#If we hit button 27, end the program
			if not GPIO.input(27):
				print("End game")
				break
					
		except KeyboardInterrupt:
			break

	#OpenCV and PIO cleanup before program ending
	cv2.destroyAllWindows()
	capture.release()
	GPIO.cleanup()


# Run main() 
if __name__ == '__main__':
	main()





