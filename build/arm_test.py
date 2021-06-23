## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################


import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 
import os

from adafruit_servokit import ServoKit
import time

import Jetson.GPIO as GPIO

import pyrealsense2 as rs

#mode='auto'
mode='user'

GPIO.cleanup()
# Declare the GPIO settings
GPIO.setmode(GPIO.BOARD)


BS=15
LS=13
RS=11
HS=16

GPIO.setup(LS, GPIO.IN)
Lc=1-GPIO.input(LS)
GPIO.setup(RS, GPIO.IN)
Rc=1-GPIO.input(RS)

GPIO.setup(BS, GPIO.IN)
Bc=1-GPIO.input(BS)

GPIO.setup(HS, GPIO.IN)
Hc=1-GPIO.input(HS)


if (Bc==0 and Lc==0 and Rc==0 and Hc==0):
 IRflag=True
 print('IR test success!')
else:
 IRflag=False
 print('IR test failed!')


PWML = 40  # ENA - H-Bridge enable pin
IN1L = 37  # IN1 - Forward Drive
IN2L = 35  # IN2 - Reverse Drive
# Motor B, Right Side GPIO CONSTANTS
PWMR = 29  # ENB - H-Bridge enable pin
IN1R = 33  # IN1 - Forward Drive
IN2R = 31  # IN2 - Reverse Drive


def init(verbose=False):
 if verbose:
  print('starting...')
 # set up GPIO pins
 GPIO.setup(PWML, GPIO.OUT) # Connected to PWMA
 GPIO.setup(IN1L, GPIO.OUT) # Connected to AIN2
 GPIO.setup(IN2L, GPIO.OUT) # Connected to AIN1
 GPIO.setup(PWMR, GPIO.OUT) # Connected to BIN1
 GPIO.setup(IN1R, GPIO.OUT) # Connected to BIN2
 GPIO.setup(IN2R, GPIO.OUT) # Connected to PWMB

def stop(verbose=False):
 if verbose:
  print('stopping...')
 # Reset all the GPIO pins by setting them to LOW
 GPIO.output(IN1L, GPIO.LOW) # Set AIN1
 GPIO.output(IN2L, GPIO.LOW) # Set AIN2
 GPIO.output(PWML, GPIO.LOW) # Set PWMA
 GPIO.output(IN2R, GPIO.LOW) # Set BIN1
 GPIO.output(IN1R, GPIO.LOW) # Set BIN2
 GPIO.output(PWMR, GPIO.LOW) # Set PWMB

def forward(dts=1,verbose=False):
 if verbose:
  print('Drive the motor forward')
 # Motor A:
 GPIO.output(IN1L, GPIO.HIGH) # Set AIN1
 GPIO.output(IN2L, GPIO.LOW) # Set AIN2
 # Motor B:
 GPIO.output(IN1R, GPIO.HIGH) # Set BIN1
 GPIO.output(IN2R, GPIO.LOW) # Set BIN2
 
 # Set the motor speed
 # Motor A:
 GPIO.output(PWML, GPIO.HIGH) # Set PWMA
 # Motor B:
 GPIO.output(PWMR, GPIO.HIGH) # Set PWMB
 
 
 # Wait 5 seconds
 time.sleep(dts)
 stop()

def reverse(dts=1,verbose=False):
 if verbose:
  print('Drive the motor reverse')
 # Motor A:
 GPIO.output(IN1L, GPIO.LOW) # Set AIN1
 GPIO.output(IN2L, GPIO.HIGH) # Set AIN2
 # Motor B:
 GPIO.output(IN1R, GPIO.LOW) # Set BIN1
 GPIO.output(IN2R, GPIO.HIGH) # Set BIN2 
 
 # Set the motor speed
 # Motor A:
 GPIO.output(PWML, GPIO.HIGH) # Set PWMA
 # Motor B:
 GPIO.output(PWMR, GPIO.HIGH) # Set PWMB
 
 
 # Wait 5 seconds
 time.sleep(dts)
 stop()

def tRight(dts=1,verbose=False):
 if verbose:
  print('Drive the motor Right')
 # Motor A:
 GPIO.output(IN2L, GPIO.HIGH) # Set AIN1
 GPIO.output(IN1L, GPIO.LOW) # Set AIN2
 # Motor B:
 GPIO.output(IN1R, GPIO.HIGH) # Set BIN1
 GPIO.output(IN2R, GPIO.LOW) # Set BIN2
 
 # Set the motor speed
 # Motor A:
 GPIO.output(PWML, GPIO.HIGH) # Set PWMA
 # Motor B:
 GPIO.output(PWMR, GPIO.HIGH) # Set PWMB
  
 
 # Wait 5 seconds
 time.sleep(dts)
 stop()

def tLeft(dts=1,verbose=False):
 if verbose:
  print('Drive the motor Left')
 # Motor A:
 GPIO.output(IN2L, GPIO.LOW) # Set AIN1
 GPIO.output(IN1L, GPIO.HIGH) # Set AIN2
 # Motor B:
 GPIO.output(IN1R, GPIO.LOW) # Set BIN1
 GPIO.output(IN2R, GPIO.HIGH) # Set BIN2
 
 # Set the motor speed
 # Motor A:
 GPIO.output(PWML, GPIO.HIGH) # Set PWMA
 # Motor B:
 GPIO.output(PWMR, GPIO.HIGH) # Set PWMB
 
 
 # Wait 5 seconds
 time.sleep(dts)
 stop()


def reset(verbose=False):
 if verbose:
  print('reseting...')
 # Reset all the GPIO pins by setting them to LOW
 GPIO.output(IN1L, GPIO.LOW) # Set AIN1
 GPIO.output(IN2L, GPIO.LOW) # Set AIN2
 GPIO.output(PWML, GPIO.LOW) # Set PWMA
 GPIO.output(IN2R, GPIO.LOW) # Set BIN1
 GPIO.output(IN1R, GPIO.LOW) # Set BIN2
 GPIO.output(PWMR, GPIO.LOW) # Set PWMB


# init arm
print('initializing...')
kit = ServoKit(channels=16)

pilot=0

arm1=0
arm2=1
arm3=2
arm4=3

cam1=5
cam2=4

dts=0.5

def_conf=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
nav_pos=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
grab_pos=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
def_conf[pilot]=90

def_conf[arm1]=90
def_conf[arm2]=90
def_conf[arm3]=110
def_conf[arm4]=90

def_conf[cam1]=90
def_conf[cam2]=90

#camera cam1
kit.servo[cam1].set_pulse_width_range(500, 1500)
kit.servo[cam1].angle = 5
time.sleep(1)
kit.servo[cam1].angle = 175
time.sleep(1)
kit.servo[cam1].angle = def_conf[cam1]

#camera cam2
kit.servo[cam2].set_pulse_width_range(1000, 2500)
kit.servo[cam2].angle = def_conf[cam2]


step = 30
time.sleep(1)
kit.servo[arm1].set_pulse_width_range(1000, 2000)
kit.servo[arm2].set_pulse_width_range(500, 2500)
kit.servo[arm3].set_pulse_width_range(500, 2500)
kit.servo[arm4].set_pulse_width_range(950, 2050)

kit.servo[arm1].angle = def_conf[arm1]
kit.servo[arm2].angle = def_conf[arm2]
kit.servo[arm3].angle = def_conf[arm3]
kit.servo[arm4].angle = def_conf[arm4]

max_arm = [175,175,175,175]
min_arm = [5,5,5,5]
close_pos=160

nav_pos[arm1]=max_arm[arm1]
nav_pos[arm2]=min_arm[arm2]
nav_pos[arm3]=max_arm[arm3]
nav_pos[arm4]=max_arm[arm4]

grab_pos[arm1]=min_arm[arm1]
grab_pos[arm2]=max_arm[arm2]
grab_pos[arm3]=110
grab_pos[arm4]=min_arm[arm4]

check_max = False
if check_max:
 print('arm1')
 kit.servo[arm1].angle = min_arm[0]
 time.sleep(1)
 kit.servo[arm1].angle = max_arm[0]
 time.sleep(1)
 kit.servo[arm1].angle = def_conf[arm1]
 time.sleep(1)
 print('arm2')
 kit.servo[arm2].angle = min_arm[1]
 time.sleep(1)
 kit.servo[arm2].angle = max_arm[1]
 time.sleep(1)
 kit.servo[arm2].angle = def_conf[arm2]
 time.sleep(1)
 print('arm3')
 kit.servo[arm3].angle = min_arm[2]
 time.sleep(1)
 kit.servo[arm3].angle = max_arm[2]
 time.sleep(1)
 kit.servo[arm3].angle = def_conf[arm3]
 time.sleep(1)
 print('arm4')
 kit.servo[arm4].angle = min_arm[3]
 time.sleep(1)
 kit.servo[arm4].angle = max_arm[3]
 time.sleep(1)
 kit.servo[arm4].angle = def_conf[arm4]


print('grab position')
kit.servo[arm4].angle = grab_pos[arm4]
kit.servo[arm1].angle = min_arm[arm1]+5
kit.servo[arm2].angle = max_arm[arm2]-5
kit.servo[arm3].angle = grab_pos[arm3]
#input('press a enter to grab...')


time.sleep(3)

print('grabbing...')
Hc=0
while Hc==0: 
 Hc=1-GPIO.input(HS)
print('closing!')

kit.servo[arm1].angle = min_arm[arm1]
kit.servo[arm2].angle = max_arm[arm2]-25
kit.servo[arm4].angle = close_pos
time.sleep(0.75)
kit.servo[arm4].angle = close_pos
kit.servo[arm3].angle = nav_pos[arm3]
kit.servo[arm1].angle = max_arm[arm1]-10
kit.servo[arm2].angle = 55
kit.servo[arm4].angle = close_pos

input('press a enter to leave...')
kit.servo[arm1].angle = min_arm[arm1]+20
kit.servo[arm2].angle = max_arm[arm2]-20
kit.servo[arm3].angle = grab_pos[arm3]
time.sleep(0.5)
kit.servo[arm4].angle = min_arm[arm4]
time.sleep(0.5)
kit.servo[arm4].angle = grab_pos[arm4]
kit.servo[arm1].angle = grab_pos[arm1]
time.sleep(0.5)

print('navigation position')
kit.servo[arm1].angle = nav_pos[arm1]
kit.servo[arm2].angle = nav_pos[arm2]
kit.servo[arm3].angle = nav_pos[arm3]
kit.servo[arm4].angle = nav_pos[arm4]



