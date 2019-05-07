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



# Brush settings 
radius = 2
coords = [(200,120),(200,121),(200,122),(200,123)]
RED = 255,0,0
GREEN = 0,255,0
BLUE = 0,0,255
WHITE = 255,255,255
BLACK = 0,0,0

colors = [RED, GREEN, BLUE, WHITE, BLACK]

color_index = 0
curr_color = RED

# Margin settings
L_MARGIN = 10
R_MARGIN = 310
T_MARGIN = 10
B_MARGIN = 230

screen.fill(black)

'''
myfont = pygame.font.Font(None, 20)
textsurface = myfont.render('Some Text', True, WHITE)
rect = textsurface.get_rect(center=(30,30))
screen.blit(textsurface, rect) 
pygame.display.flip()
'''


# ======= SAMPLE_CAM3.PY CODE ============ #

# Setup video capture variables 
capture = cv2.VideoCapture(0)
hand_hist = None
draw = False
is_hand_hist_created = False


traverse_point  = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

# Current and previous point trackers
prev = None
curr = None
prev_dot = None
curr_dot = None
draw_thresh = 20


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
	
	
# ======================== TRACE HAND ======================== #

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
       # print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        
       
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


def interpolate(prev_coord, curr_coord):
    
    if (prev_coord is not None) and (curr_coord is not None):
	prev_scaled = 320 - int(prev_coord[0]/2), int(prev_coord[1]/2)
	curr_scaled = 320 - int(curr_coord[0]/2), int(curr_coord[1]/2)
	
	pygame.draw.line(screen, curr_color, prev_scaled, curr_scaled, radius*2)
	pygame.display.flip()


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
		      
def change_color():
    global curr_color, color_index
    color_index +=1 
    if color_index >= len(colors):
	color_index = 0
    curr_color = colors[color_index]
    print(curr_color)
    
def change_radius(up_or_down):
    global radius
    if up_or_down:
	radius+=1
    else:
	radius-=1
    
    
# ==================================================================== #
# |                      MULTICORE FUNCTIONS 			     | #
# ==================================================================== #
	
# Master process function	
def master_process(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start):
    global hand_hist
    global draw 
    global is_hand_hist_created
    global capture
    
    # Time variables
    last_receive_time = 0
    start_time_ms = 0
    contour_read = False
    start_time = 0
    start_datetime = datetime.now()
    
    # Current and previous point trackers
    global prev 
    global curr 
    global prev_dot 
    global curr_dot
    global draw_thresh 
	        
    # Pygame screen setup (black background)
    screen.fill(black)
    
    while (run_flag.value and capture.isOpened()):
	try:
		    
	    # Wait for keypress 
	    pressed_key = cv2.waitKey(1)
	    _, frame = capture.read()

	    # Press z to create hand histogram
	    if pressed_key & 0xFF == ord('z'):
		is_hand_hist_created = True
		hand_hist = hand_histogram(frame)

	    # Check if time since last send to queue exceeds 30ms
	    current_time = datetime.now()
	    time_dif = current_time - start_datetime
	    time_dif_ms = time_dif.total_seconds()*1000
	    
	    # If d is pressed, draw
	    if pressed_key & 0xFF == ord("d"): 
		print("Draw button pressed!")
		draw = not draw
	    
	    # Place frame in queue only if time has exceeded 30ms, and 
	    # there are fewer than 4 frames in each queue.
	    if ((send_frame_queue.qsize() < 4) and (send_hist_queue.qsize() < 4)):
		start_datetime = current_time 		# Update last send to queue time
		if is_hand_hist_created:
		    #print("Hist created")
		    #frame_data = (frame, hand_hist)	# Frame and hand histogram data
		    send_frame_queue.put(frame) 	# Put frame in queue
		    send_hist_queue.put(hand_hist)	# Put histogram in queue
		

		    # Check if receive_point_queue is not empty
		    if (not receive_point_queue.empty()):
			last_receive_time = time.time()
			far_point = receive_point_queue.get()	# Extract far_point data
			
		    
				
			if far_point is not None:
			    curr = far_point
			   
			    # Draw dots and interpolate between them 
			    if draw:
				if l2_distance(prev, curr) <= draw_thresh:
				    prev_dot = curr_dot
				    curr_dot = far_point
				    draw_dot(far_point)
				    interpolate(prev_dot,curr_dot)
				else:
				    interpolate(prev_dot, curr_dot)
		
			    # Update the current and previous points
			    if prev is None:
				prev = far_point
			    else:
				prev = curr
			    
			    
		else:
		    frame = draw_rect(frame)

		cv2.imshow("Live Feed", rescale_frame(frame))
	    
	    # Break
	    if pressed_key & 0xFF == ord('q'):
		run_flag.value = 0
		#break
	    
	    # Change color when button 17 pressed	
	    if not GPIO.input(17):
		change_color()
	    
	    # Increase size of brush when button 22 pressed
	    if not GPIO.input(22):
		change_radius(True)
	    
	    # Decrease size of brush when button 23 pressed
	    if not GPIO.input(23):
		change_radius(False)
	    
	    # Bailout button 27   
	    if not GPIO.input(27):
		run_flag.value = 0
		print("Quit")
		#break
		
	except KeyboardInterrupt:
	    run_flag.value = 0
	    
	   
	    
# Function for worker process 1    
def process_1(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start):
    while (run_flag.value and capture.isOpened()):
	startTime = datetime.now()
	startTime_ms = startTime.second*1000 + startTime.microsecond/1000
	
	# If the frame queue is not empty and it is worker 1's turn
	if ((not send_frame_queue.empty()) and (not send_hist_queue.empty()) and (p_start.value == 1)):
	    print("Processor 1 working")
	    mask = send_frame_queue.get()	# Grab the frame
	    hist = send_hist_queue.get()	# Grab histogram
	    p_start.value = 2 			# Change to worker 2's turn
	    
	    # Calculate far_point
	    far_point = manage_image_opr(mask, hist)
	    
	    # Obtain center and max_contour
	    ctr, mc = get_centroid(mask, hist)
	    
	    # Put far_point back into queue
	    receive_point_queue.put(far_point)
	
	else:
	    print("Processor 1 did not receive any information, sleeping for 30ms zzz")
	    time.sleep(0.03)
	
	currentTime = datetime.now()
	currentTime_ms = currentTime.second*1000 + currentTime.microsecond/1000

    print("Quitting Processor 1")	
    
# Function for worker process 2    
def process_2(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start):
    while (run_flag.value and capture.isOpened()):
	startTime = datetime.now()
	startTime_ms = startTime.second*1000 + startTime.microsecond/1000
	
	# If the frame queue is not empty and it is worker 1's turn
	if ((not send_frame_queue.empty()) and (not send_hist_queue.empty()) and (p_start.value == 2)):
	    print("Processor 2 working")
	    mask = send_frame_queue.get()	# Grab the frame
	    hist = send_hist_queue.get()	# Grab histogram
	    p_start.value = 3 			# Change to worker 2's turn
	    
	    # Calculate far_point
	    far_point = manage_image_opr(mask, hist)
	    
	    # Obtain center and max_contour
	    ctr, mc = get_centroid(mask, hist)
	    
	    # Put far_point back into queue
	    receive_point_queue.put(far_point)
	
	else:
	    print("Processor 2 did not receive any information, sleeping for 30ms zzz")
	    time.sleep(0.03)
	
	currentTime = datetime.now()
	currentTime_ms = currentTime.second*1000 + currentTime.microsecond/1000
	
    print("Quitting Processor 2")
    
# Function for worker process 3       
def process_3(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start):
    while (run_flag.value and capture.isOpened()):
	startTime = datetime.now()
	startTime_ms = startTime.second*1000 + startTime.microsecond/1000
	
	# If the frame queue is not empty and it is worker 1's turn
	if ((not send_frame_queue.empty()) and (not send_hist_queue.empty()) and (p_start.value == 3)):
	    print("Processor 3 working")
	    mask = send_frame_queue.get()	# Grab the frame
	    hist = send_hist_queue.get()	# Grab histogram
	    p_start.value = 1 			# Change to worker 2's turn
	    
	    # Calculate far_point
	    far_point = manage_image_opr(mask, hist)
	    
	    # Obtain center and max_contour
	    ctr, mc = get_centroid(mask, hist)
	    
	    # Put far_point back into queue
	    receive_point_queue.put(far_point)
	
	else:
	    print("Processor 3 did not receive any information, sleeping for 30ms zzz")
	    time.sleep(0.03)
	
	currentTime = datetime.now()
	currentTime_ms = currentTime.second*1000 + currentTime.microsecond/1000
    
    print("Quitting Processor 3")
    

# ================== MAIN ================== #
def main():
    # Set run_flag to safely exit all processes
    run_flag = Value('i', 1)
    
    # p_start_turn determines worker processing order
    p_start = Value('i', 1)
    
    # Set up queues
    send_frame_queue = Queue()
    send_hist_queue = Queue()
    receive_point_queue = Queue()
    
    
    # Set up four processes: 1 master process, 3 worker processes
    p0 = Process(target=master_process, args=(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start))
    p1 = Process(target=process_1, args=(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start))
    p2 = Process(target=process_2, args=(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start))
    p3 = Process(target=process_3, args=(run_flag, send_frame_queue, send_hist_queue, receive_point_queue, p_start))
    
    # Start and join processes
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    
    p0.join()
    p1.join()
    p2.join()
    p3.join()

    cv2.destroyAllWindows()
    GPIO.cleanup()
    capture.release()
    


# Run main() 
if __name__ == '__main__':
    main()





