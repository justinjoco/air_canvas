"""
bounce.py
Justin Joco(jaj263), Stephanie Lin(scl97)
ECE 5725 Lab 2
"""


import sys
import pygame
import time
import RPi.GPIO as GPIO
import os


#Set environment variables
os.putenv('SDL_VIDEODRIVER','fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')

#Initialize game
pygame.init()

#
size = width, height = 320, 240


#speed = [5,5]
black = 0,0,0

screen = pygame.display.set_mode(size)

#ball = pygame.image.load('bouncy_ball.png')
#ballrect = ball.get_rect()

time_end = time.time() + 10



radius = 2
coords = [(200,120),(200,121),(200,122),(200,123)]
RED = 255,0,0

screen.fill(black)
for i in range(100,200,1):
    time.sleep(.02)
    pygame.draw.circle(screen,RED, (200, i), radius)
    pygame.display.flip()

for j in range(i,100,-1):
    time.sleep(.02)
    pygame.draw.circle(screen,RED, (j, i), radius)
    pygame.display.flip()

while time.time()<time_end:
    time.sleep(.05)
     
    
    try:
        print("filler")
        '''    
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        ballrect = ballrect.move(speed)
        if ballrect.left < 0 or ballrect.right > width:
            speed[0] = -speed[0]
        if ballrect.top < 0 or ballrect.bottom > height:
            speed[1] = -speed[1]
        


               #screen.blit(ball, ballrect)
        pygame.display.flip()
        '''
    except KeyboardInterrupt:
        break

