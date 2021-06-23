## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import random
import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 
import os
import json

from adafruit_servokit import ServoKit
import time

import Jetson.GPIO as GPIO

from multiprocessing import Process
from playsound import playsound

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

GPIO.setup(BS, GPIO.IN)
Bc=1-GPIO.input(BS)

GPIO.setup(LS, GPIO.IN)
Lc=1-GPIO.input(LS)
GPIO.setup(RS, GPIO.IN)
Rc=1-GPIO.input(RS)

GPIO.setup(HS, GPIO.IN)
Hc=1-GPIO.input(HS)


if (Bc==0 and Lc==0 and Rc==0 and Hc==0):
 IRflag=True
 print('IR test success!')
else:
 IRflag=False
 print('IR test failed!')

arm_cam = cv2.VideoCapture("/dev/video0")

if arm_cam.isOpened(): # try to get the first frame
   rval, arm_cam_test_frame = arm_cam.read()
   print('arm camera test success!')
else:
   rval = False
   print('arm camera test failed!')

with open('CMDS.json') as json_file:
    CMDS = json.load(json_file)
print('commands loaded!')

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
def_conf[arm3]=90
def_conf[arm4]=90

def_conf[cam1]=90
def_conf[cam2]=90

print('motor test...')
init()
tLeft(dts=0.25)
tRight(dts=0.25)
stop()


timeStamp=time.time()
fpsFilt=0
#net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)
dispW=1280
dispH=720
#dispW=1024
#dispH=600
flip=2
font=cv2.FONT_HERSHEY_SIMPLEX

# Gstreamer code for improvded Raspberry Pi Camera Quality
#camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink'
#cam=cv2.VideoCapture(camSet)
#cam=jetson.utils.gstCamera(dispW,dispH,'0')

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

nav_pos[arm1]=max_arm[arm1]
nav_pos[arm2]=min_arm[arm2]
nav_pos[arm3]=max_arm[arm3]
nav_pos[arm4]=max_arm[arm4]

grab_pos[arm1]=min_arm[arm1]+5
grab_pos[arm2]=max_arm[arm2]-5
grab_pos[arm3]=110
grab_pos[arm4]=min_arm[arm4]


print('navigation position')
kit.servo[arm1].angle = nav_pos[arm1]
kit.servo[arm2].angle = nav_pos[arm2]
kit.servo[arm3].angle = nav_pos[arm3]
kit.servo[arm4].angle = nav_pos[arm4]

cPhi = def_conf[cam1]
hPhi = def_conf[cam2]
sPhi = 2
s2Phi = 2
cdval = 6
csval = 10
cs = csval

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)

Rc=0
Lc=0
Bc=0
sf=1/1000
fpsFilt=10
cooldown=0
mstep=1

lower_yellow = np.array([15, 15, 15]) 
upper_yellow = np.array([45, 255, 255])

lower_red = np.array([45,105,55])
upper_red = np.array([90,255,245])
nframe=0 

scroll=0
scroll_step=0.25
scroll_max=2.5
controller =  True
tracking = False
tracker = cv2.TrackerMOSSE_create()
begin_frame = True
#clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
flag_reset = True
rho_reset=0.5
Lcounter=0
Lcmax=3
Rcounter=0
Rcmax=3
text = 'hello'
Cup = 175
Cdwn = 25
counter = 0
dts = 0
ball_dist = 0.22

track_human = True
track_ball = False
tgt_dist = 0.7
tgt_ratio = 1.5

close_pos=160
grabbing = False
GRABINT = False
SEARCHINT = False
grab_counter=0
wait_counter = -1
track_counter = -1
SAMBAINT=False
samba_var=0
samba_dir=1
samba_amp=1

try:
    while True:
        if track_human:
            Cup = 175
            Cdwn = 25

        if track_ball and not(SEARCHINT):
            Cup = 135
            Cdwn = 5

        if track_ball and SEARCHINT:
            Cup = 50
            Cdwn = 5

        counter+=1
        f=open('queque_number.txt','r')
        qn = int(f.read())
        f.close()
        #print('queque:',qn)
        try:
            f=open('queque/'+str(qn+1)+'.txt','r')
            temp = f.read()
            f.close()
            if len(temp)>0:
                    text = temp
                    f=open('queque_number.txt','w')
                    f.write(str(qn+1))
                    f.close()
            os.remove('queque/'+str(qn+1)+'.txt')
            dta = 0.2
            if text.lower() in CMDS['LEFT']['QUERY']:
                    tLeft(dta*2)
            if text.lower() in CMDS['RIGHT']['QUERY']:
                    tRight(dta*2)
            if text.lower() in CMDS['FWD']['QUERY']:
                    forward(dta*3)
            if text.lower() in CMDS['BACK']['QUERY']:
                    reverse(dta*3)
            if text.lower() in CMDS['2xFWD']['QUERY']:
                    forward(dta*6)
            if text.lower() in CMDS['2xBACK']['QUERY']:
                    reverse(dta*6)
            if text.lower() in CMDS['HLEFT']['QUERY']:
                    tLeft(dts)
            if  text.lower() in CMDS['HRIGHT']['QUERY']:
                    tRight(dta)
            if  text.lower() in CMDS['2xLEFT']['QUERY']:
                    tLeft(dta*4)
            if  text.lower() in CMDS['2xRIGHT']['QUERY']:
                    tRight(dta*4)
            if  text.lower() in CMDS['HFWD']['QUERY']:
                    forward(dta*1.5)
            if  text.lower() in CMDS['HBACK']['QUERY']:
                    reverse(dta*1.5)
            if  text.lower() in CMDS['GET_CLOSE']['QUERY']:
                    tgt_dist=0.4
            if text.lower() in CMDS['GET_AWAY']['QUERY']:
                    tgt_dist=0.7
            if  text.lower() in CMDS['QUIT']['QUERY']:
                    break
            if  text.lower() in CMDS['AUTO']['QUERY']:
                    mode='auto'
                    cooldown=0
                    cs=csval
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
            if  text.lower() in CMDS['USER']['QUERY']:
                    mode='user'
            if  text.lower() in CMDS['LOOK_LEFT']['QUERY']:
                    hPhi=5
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
            if text.lower() in CMDS['LOOK_RIGHT']['QUERY']:
                    hPhi=175
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
            if text.lower() in CMDS['LOOK_DOWN']['QUERY']:
                    cPhi=Cdwn+10
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
            if text.lower() in CMDS['LOOK_FRONT']['QUERY']:
                    cPhi=Cdwn+10
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
            if text.lower() in CMDS['LOOK_UP']['QUERY']:
                    cPhi=Cup-10
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
            #miscellanea
            if text.lower() in CMDS['HELLO']['QUERY']:
                    tLeft(dta*0.9)
                    tRight(dta*0.9)
                    tLeft(dta*0.7)
                    tRight(dta*0.7)

            if text.lower() in CMDS['BYE']['QUERY']:
                    tLeft(dta*0.9)
                    tRight(dta*0.9)
                    tLeft(dta*0.7)
                    tRight(dta*0.7)
                    time.sleep(0.25)
                    break

            if text.lower() in CMDS['TRACK_ME']['QUERY']:
                    track_human = True
                    track_ball = False
                    tgt_dist = 0.7
                    tgt_ratio = 1.5

            if text.lower() in CMDS['TRACK_BALL']['QUERY']:
                    track_human = False
                    track_ball = True
                    tgt_dist = ball_dist
                    tgt_ratio = 1.25

            if text.lower() in CMDS['GRAB_BALL']['QUERY']:
                    kit.servo[arm4].angle = grab_pos[arm4]
                    kit.servo[arm1].angle = min_arm[arm1]+5
                    kit.servo[arm2].angle = max_arm[arm2]-20
                    kit.servo[arm3].angle = grab_pos[arm3]

            if text.lower() in CMDS['GET_BALL']['QUERY']:                    
                    track_human = False
                    track_ball = True
                    tgt_dist = ball_dist
                    tgt_ratio = 1.25
                    kit.servo[arm4].angle = grab_pos[arm4]
                    kit.servo[arm1].angle = min_arm[arm1]+1
                    kit.servo[arm2].angle = max_arm[arm2]-15
                    kit.servo[arm3].angle = grab_pos[arm3]
                    mode='auto'
                    cooldown=0
                    grab_counter=0
                    cs=csval
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
                    grabbing = True
                    SEARCHINT = False

            if text.lower() in CMDS['FIND_BALL']['QUERY']:
                    track_human = False
                    track_ball = True
                    tgt_dist = ball_dist
                    tgt_ratio = 1.25
                    kit.servo[arm1].angle = nav_pos[arm1]
                    kit.servo[arm2].angle = nav_pos[arm2]
                    kit.servo[arm3].angle = nav_pos[arm3]
                    kit.servo[arm4].angle = nav_pos[arm4]
                    mode='user'
                    SEARCHINT = True
                    GRABINT = False
                    grabbing = False
                    cooldown=0
                    cs=csval
                    old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
                    grabbing = True
                    print('search ball')

            if text.lower() in CMDS['RELAX']['QUERY']:
                    kit.servo[arm1].angle = nav_pos[arm1]
                    kit.servo[arm2].angle = nav_pos[arm2]
                    kit.servo[arm3].angle = nav_pos[arm3]
                    kit.servo[arm4].angle = nav_pos[arm4]
                    track_human = True
                    track_ball = False
                    tgt_dist = 0.7
                    tgt_ratio = 1.5

            if text.lower() in CMDS['CLOSE_HAND']['QUERY']:
                    forward(dta*1.25)
                    kit.servo[arm4].angle = close_pos
                    time.sleep(0.75)
                    kit.servo[arm1].angle = max_arm[arm1]
                    kit.servo[arm2].angle = 45

            if text.lower() in CMDS['LEAVE']['QUERY']:
                    kit.servo[arm1].angle = min_arm[arm1]+20
                    kit.servo[arm2].angle = max_arm[arm2]-20
                    kit.servo[arm3].angle = grab_pos[arm3]
                    time.sleep(0.5)
                    kit.servo[arm4].angle = min_arm[arm4]
                    time.sleep(0.5)
                    kit.servo[arm4].angle = grab_pos[arm4]
                    kit.servo[arm1].angle = grab_pos[arm1]
                    time.sleep(0.5)

            if text.lower() in CMDS['LAUNCH_BALL']['QUERY']:
                    kit.servo[arm1].angle = min_arm[arm1]
                    kit.servo[arm2].angle = max_arm[arm2]-25
                    kit.servo[arm4].angle = close_pos
                    kit.servo[arm3].angle = grab_pos[arm3]
                    time.sleep(0.3)

                    run_up=0.5
                    reverse(run_up)

                    kit.servo[arm1].angle = nav_pos[arm1]
                    kit.servo[arm2].angle = max_arm[arm2]-5

                    time.sleep(0.15)
                    forward(run_up)
                    kit.servo[arm1].angle = nav_pos[arm1]-55
                    kit.servo[arm4].angle = grab_pos[arm4]
                    stop()
                    time.sleep(0.5)

                    print('navigation position')
                    kit.servo[arm1].angle = nav_pos[arm1]
                    kit.servo[arm2].angle = nav_pos[arm2]
                    kit.servo[arm3].angle = nav_pos[arm3]
                    kit.servo[arm4].angle = nav_pos[arm4]

            if text.lower() in CMDS['PLAY_SAMBA']['QUERY']:
                    if os.path.exists('queque/samba_stop.txt'):
                       os.remove('queque/samba_stop.txt')
                    f = open('queque/samba_start.txt','w+')
                    f.write('samba_short2.mp3')
                    f.close()
                    SAMBAINT=True
                    samba_var=1
                    samba_amp=1

            if text.lower() in CMDS['STOP_SAMBA']['QUERY']:
                    f = open('queque/samba_alt.txt','w+')
                    f.write('terminate')
                    f.close()
                    SAMBAINT=False
                    samba_var=0
                    samba_amp=1
                    

        except:
            void=[]
        

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        if GRABINT:
            rval, color_image = arm_cam.read()
        else:
            color_image = np.asanyarray(color_frame.get_data())

        #lab = cv2.cvtColor(color_image, cv2.COLOR_RGB2LAB)
        #lab_planes = cv2.split(lab)
        #lab_planes[0] = cv2.equalizeHist(lab_planes[0])
        #lab = cv2.merge(lab_planes)
        #color_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV) 
        # preparing the mask to overlay 
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        red_mask = cv2.erode(red_mask,kernel,iterations = 3)
        red_mask = cv2.dilate(red_mask,kernel,iterations = 2)
        bkp_mask = 1*red_mask
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 1)

        # Connected components with stats.
        nb_components, cc_output, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=4)

        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        #try:
        if track_ball:
                max_label=-1
                tmin=1000
                mside = 30
                #print('components:',nb_components)
                if nb_components>1:
                   for cit in range(1, nb_components):
                      if np.sum(1*(cc_output==cit), axis = None)>mside*mside:
                         tmd = depth_image[cc_output==cit]
                         tdist=sf*np.mean(tmd[(sf*tmd)>0.05], axis = None)
                         if tdist<tmin:
                            max_label=cit
                            tmin=tdist
                            ys, xs = np.where(cc_output==cit)
                            red_centroid = [np.mean(xs, axis = None), np.mean(ys, axis=None)]

                   #print('tmin:',tmin)

                   if tmin<1000:                   
                      red_mask = (255*(cc_output==max_label)).astype(np.uint8)
                      temp_centroid = cc_centroids[max_label]
                      #print(red_centroid)
                      red_contours,_ = cv2.findContours(red_mask, 1, 1) 
                      red_rect = cv2.minAreaRect(red_contours[0]) 

                      red_box = cv2.boxPoints(red_rect)
                      red_box = np.int0(red_box) #turn into ints
                   else:
                      red_centroid = []
                else:
                   red_centroid = []
        #except:
                #print('error!')
                #red_centroid = []

            

        #tracking_frame = cv2.resize(color_image, (NW,NH))
        tracking_frame = 1*color_image
        (H, W) = tracking_frame.shape[:2]

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #depth_colormap = depth_image
        height=color_image.shape[0]
        width=color_image.shape[1]
        bnds=color_image.shape
        frame=jetson.utils.cudaFromNumpy(color_image)


        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        detections=net.Detect(frame, width, height)
        pos=[0,0]
        d=-1
        bdist=-1
        trig_dist = -1
        initBB=[]

        
        if track_ball:
            #color_image[red_mask==0] = [0,0,0]
            void=[]
        #cv2.rectangle(color_image, (int(ry), int(rx)), (int(ry+rh), int(rx+rw)), (255, 255, 0), 2)
        if track_ball and len(red_centroid)==2:
            cv2.drawContours(color_image,[red_box],0,(0,0,255),10)

        for detect in detections:
            #print(detect)
            ID=detect.ClassID
            item=net.GetClassDesc(ID)
            conf=detect.Confidence
            top=detect.Top
            left=detect.Left
            bottom=detect.Bottom
            right=detect.Right
            #print(item,top,left,bottom,right)
            cv2.rectangle(color_image, (int(left-6), int(top)), (int(right+6), int(bottom)), (0, 0, 255), 2)
            cv2.rectangle(color_image, (int(left-6), int(bottom - 35)), (int(right+6), int(bottom)), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(color_image, item+' '+str(round(conf*100))+'%', (int(left -6 + 6), int(bottom - 10)), font, 0.7, (255, 255, 255), 1)
            if track_human and not(SAMBAINT) and item=='person' and conf>0.45:
                x=int(left)
                y=int(top)
                w=int(right-left)
                h=int(bottom-top)
                rect = [int(top), int(left), int(bottom), int(right)] # top, left, bottom, right
                rmask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
                cv2.rectangle(rmask, (rect[0], rect[1]), (rect[2], rect[3]), 255, cv2.FILLED)
                rect_mean = cv2.mean(mask, rmask)[0]
                med = sf*np.median(depth_image[y:y+h,x:x+w],axis=None)
                #print('median:',med)
                mask = (sf*depth_image<(med+0.1))*(sf*depth_image>(med-0.1)).astype(np.uint8)
                #tdist = sf*cv2.mean(depth_image, mask)[0]
                tdist = sf*(0.25*np.median(depth_image[mask>0], axis = None)+0.5*np.mean(depth_image[mask>0], axis = None)+0.25*depth_image[mask>0].min())
                #print('d:',d)
                td = 0.7*rect_mean/15.0+0.3*conf
                if conf>0.6 and td>d:
                    d=td
                    pos=[x+w/2,y+h/2]
                    yrect=rect
                    bconf=conf
                    bdist=tdist
                    initBB=(x,y,w,h)
                    trig_dist = bdist*np.cos(np.pi*(cPhi-90.0)/180.0)
        if track_human and d>0.7:
            #tracker = cv2.TrackerMedianFlow_create()
            tracker = cv2.TrackerMOSSE_create()
            tracker.init(tracking_frame, initBB)
            tracking = True
            #print('starting box:',initBB)
        
        if track_human and tracking:
        #if True:
            (success, box) = tracker.update(tracking_frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                pos=[x+w/2,y+h/2]
                #print('green BBox:',(x, y, w, h))
                cv2.rectangle(color_image, (int(x+6), int(y-6)), (int(x+w-6), int(y+h-6)), (255, 0, 0), 2)
            else: 
                tracking = False

        if track_human and d>0.5:
            flag_reset = True                    
            cooldown=cdval
            top=yrect[0]
            left=yrect[1]
            bottom=yrect[2]
            right=yrect[3]
            cv2.rectangle(color_image, (int(left+3), int(top+3)), (int(right-3), int(bottom-38)), (0, 225, 0), 3)
            cv2.rectangle(depth_image, (int(left+3), int(top+3)), (int(right-3), int(bottom-38)), (0, 0, 0), 3)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if track_ball and len(red_centroid)==2:
            pos = red_centroid
            bdist = sf*(0.25*np.median(depth_image[red_mask>0], axis = None)+0.5*np.mean(depth_image[red_mask>0], axis = None)+0.25*depth_image[red_mask>0].min())
            d=1

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation = cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        if begin_frame:
            #fourcc = cv2.VideoWriter_fourcc(*'MP42')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('sequence.mp4', fourcc, float(5), (color_image.shape[1], color_image.shape[0]))
            begin_frame = False
        out_frame = 1*color_image
        cv2.putText(out_frame,str(round(fpsFilt,1))+' fps, dist '+str(round(bdist,2))+'m, mode '+mode, (0,30),font,1,(0,0,255),2)
        cv2.putText(out_frame, text, (0,60),font,1,(0,0,255),2)
        out.write(out_frame)
        flag_camera = False
        if d==-1:
            if cooldown>0:
                pos=old_pos
                cooldown-=1
            else: 
                if flag_reset:
                     cPhi = rho_reset*100+(1-rho_reset)*cPhi
                     flag_reset = False
                cPhi += 2*sPhi
                hPhi += 2*s2Phi
    
        if d>-1:
       
            if pos[1]<bnds[0]*0.4:
                cPhi+=3*abs(sPhi)
            if pos[1]>bnds[0]*0.6:
                cPhi-=3*abs(sPhi)

            if pos[0]<bnds[1]*0.4 and not(GRABINT):
                hPhi+=3*abs(s2Phi)
                flag_camera = True
            if pos[0]>bnds[1]*0.6 and not(GRABINT):
                hPhi-=3*abs(s2Phi)
                flag_camera = True

            if pos[0]<bnds[1]*0.45 and GRABINT:
                tRight(dts*0.4)
                turn_flag = True
                Rcounter+=1
            if pos[0]>bnds[1]*0.55 and GRABINT:
                tLeft(dts*0.4)
                turn_flag = True
                Lcounter+=1

        if SAMBAINT:
            if os.path.exists('queque/samba_stop.txt') and not(os.path.exists('queque/samba_start.txt')):
               SAMBAINT=False
               samba_var=0
               samba_amp=1

            if samba_var==1:
               tLeft(dta*0.5*samba_amp)
               samba_var=0
               samba_dir=-1

            if samba_var==-1:
               tRight(dta*0.5*samba_amp)
               samba_var=0
               samba_dir=1

            if samba_var==0:
               if samba_dir==-1:
                  samba_var=-1
                  samba_amp=(random.random()+1)/2
               if samba_dir==1:
                  samba_var=1

        scroll+=scroll_step
        if abs(scroll)>scroll_max:
            scroll_step=-scroll_step
        hPhi+=scroll_step
        if track_human:
            cPhi+=0.5*abs(sPhi)


    
        if cPhi>=Cup:
            sPhi=-sPhi
            cPhi=Cup-7
        
        if cPhi<=Cdwn:
            sPhi=-sPhi
            cPhi=Cdwn+7

        if hPhi>=175:
            s2Phi=-s2Phi
            hPhi=170
        
        if hPhi<=5:
            s2Phi=-s2Phi
            hPhi=10

        collision_flag=False
        if mode=='auto':
            dist=1000
            if IRflag:
                Bc=1-GPIO.input(BS)
                Lc=1-GPIO.input(LS)
                Rc=1-GPIO.input(RS)
            else:
                Bc=0
                Lc=0
                Rc=0
            dsize=depth_image.shape
            Lsd = depth_image[0:int(dsize[0]/2),:]
            #print(Lsd[(sf*Lsd)>0.05].min())
            Lc+=1*((0.25*np.median(Lsd[(sf*Lsd)>0.05],axis=None) +0.75*np.mean(Lsd[(sf*Lsd)>0.05],axis=None))*sf<0.1)
            Rsd = depth_image[int(dsize[0]/2):,:]
            #print(Rsd.min())
            Rc+=1*((0.25*np.median(Rsd[(sf*Rsd)>0.05],axis=None)+0.75*np.mean(Rsd[(sf*Rsd)>0.05],axis=None))*sf<0.1 )
            if not(GRABINT) and not(SEARCHINT) and (Lc>=1 or Rc>=1 or Bc>=1):
                if Lc>=1 and Rc>=1:
                   reverse(dts*0.7)
                   collision_flag=True
                if Lc>=1 and Rc==0 and Lcounter<=1.5*Lcmax:
                   reverse(0.7*dts)
                   tLeft(0.7*dts)
                   collision_flag=True
                   Lcounter+=1
                if Lc==0 and Rc>=1 and Rcounter<=1.5*Rcmax:
                   reverse(0.7*dts)
                   tRight(0.7*dts)
                   collision_flag=True
                   Rcounter+=1
                if Bc>=1:
                   forward(dts)
                   collision_flag=True


        if track_human:
            mr_rad=70
        if track_ball:
            mr_rad=80
        turn_flag = False
        if hPhi>=175-mr_rad and not(GRABINT):
            if mode=='auto' and nframe%mstep==0 and not(collision_flag) and Rcounter<=Rcmax and d>-1:
               tRight(dts*0.4)
               turn_flag = True
               Rcounter+=1
            #if cooldown<=0:
               #hPhi-=10
               #s2Phi=-abs(s2Phi)
        
        if hPhi<=5+mr_rad and not(GRABINT):
            if mode=='auto' and nframe%mstep==0 and not(collision_flag) and Lcounter<=Lcmax and d>-1: 
               tLeft(dts*0.4)
               turn_flag = True
               Lcounter+=1
            #if cooldown<=0:
               #hPhi+=10
               #s2Phi=abs(s2Phi)


        if mode=='auto':
            if not(turn_flag) and not(Lc>=1 or Rc>=1 or Bc>=1):
                if cs>0:
                   cs-=1
                #if dist<40 and dist>5 and cs==0:
                if d>0.3 and bdist<tgt_dist and nframe%mstep==0 and not(collision_flag):
                    if track_human:
                       reverse(dts*1.2)
                    if track_ball:
                       reverse(dts*0.9)
                if not(GRABINT) and not(SEARCHINT) and d>0.3 and bdist>(tgt_dist*tgt_ratio) and nframe%mstep==0 and not(collision_flag):
                    if track_human:
                       forward(dts*1.2)
                    if track_ball:
                       forward(dts*0.6)

                if track_ball and grabbing and bdist<1.2*(tgt_dist*tgt_ratio) and bdist>0.1*tgt_dist: #and abs(hPhi-90)<30:
                    grab_counter+=1
                else:
                    grab_counter=max(0,grab_counter-0.5)

                if grab_counter>3:
                    GRABINT = True                    
                    #forward(dta*1.0)
                    kit.servo[arm4].angle = grab_pos[arm4]+20
                    kit.servo[arm1].angle = min_arm[arm1]+5
                    kit.servo[arm2].angle = max_arm[arm2]-5
                    kit.servo[arm3].angle = grab_pos[arm3]
                    Hc=0
                    grab_counter=0
                    wait_counter=1
                    print('grabbing...')
                    #input('press a enter to grab...')

                if wait_counter>0:
                    wait_counter+=1

                if wait_counter>3:
                    wait_counter=0
                   

                if GRABINT and wait_counter==0:                    
                    Hc=1-GPIO.input(HS)
                    if Hc==0:
                      forward(dts*0.7)
                      stop()
                      print('waiting...')
                    else:
                     print('closing!')
                     stop()
                     kit.servo[arm4].angle = close_pos
                     kit.servo[arm1].angle = min_arm[arm1]
                     kit.servo[arm2].angle = max_arm[arm2]-25
                     forward(dta*1.5)
                     kit.servo[arm4].angle = close_pos
                     time.sleep(dts)
                     kit.servo[arm4].angle = close_pos
                     kit.servo[arm3].angle = nav_pos[arm3]
                     kit.servo[arm1].angle = max_arm[arm1]-10
                     kit.servo[arm2].angle = 55
                     kit.servo[arm4].angle = close_pos
                     Hc=0
                     GRABINT = False
                     track_human = True
                     track_ball = False
                     tgt_dist = 0.7
                     tgt_ratio = 1.5
                     wait_counter = -1


    

        if track_ball and SEARCHINT:
            if bdist==-1:
               track_counter=min(track_counter+1,3)
            else:
               track_counter=max(track_counter-0.5,0)

            if track_counter>=3:
               tRight(dts*0.4)

        old_pos=pos
        nframe+=1

        if cPhi>=Cup:
            sPhi=-sPhi
            cPhi=Cup-7
        
        if cPhi<=Cdwn:
            sPhi=-sPhi
            cPhi=Cdwn+7
    
        if not(GRABINT) and not(SAMBAINT):
            kit.servo[cam1].angle = cPhi
            kit.servo[cam2].angle = hPhi

        if GRABINT or SAMBAINT:
            kit.servo[cam1].angle = def_conf[cam1]
            kit.servo[cam2].angle = def_conf[cam2]

        #display.RenderOnce(img,width,height)
        dt=time.time()-timeStamp
        timeStamp=time.time()
        fps=1/dt
        fpsFilt=.9*fpsFilt + .1*fps
        #dts=1.0*min(1/fpsFilt,1/5)
        if track_human:
            dts=0.25
        if track_ball:
            if bdist>0 and bdist<tgt_dist*tgt_ratio:
            	dts = 0.075*(bdist+0.5)/(tgt_dist+0.5)
            else:
            	dts = 0.075
        #print(str(round(fps,1))+' fps')
        cv2.putText(images,str(round(fpsFilt,1))+' fps, dist '+str(round(bdist,2))+'m, mode '+mode+', '+text,(0,30),font,1,(0,0,255),2)
        cv2.imshow('detCam',images)
        cv2.moveWindow('detCam',0,0)
        key=cv2.waitKey(1)
        if key==ord('h'): #x key
           controller = True
        if key==ord('y'):
           controller = False
        if (key==ord('k') and controller) or (key==ord('z') and not(controller)):
           break
        if (key==ord('j') and controller) or (key==ord('u') and not(controller)):
           mode='user'
        if (key==ord('g') and controller) or (key==ord('i') and not(controller)):
           mode='auto'
           cooldown=0
           cs=csval
           old_pos=[int(bnds[1]*0.5),int(bnds[0]*0.5)]
           nframe=0
        if (key==ord('c') and controller) or (key==ord('w') and not(controller)):
           forward(dts)
        if (key==ord('d') and controller) or (key==ord('s') and not(controller)):
           reverse(dts)
        if (key==ord('e') and controller) or (key==ord('a') and not(controller)):
           if mode=='user' and Lcounter<=Lcmax:           
            tLeft(dts)
            Lcounter+=1
           else:
            if Lcounter<=Lcmax:
               tLeft(0.75*dts)
               Lcounter+=1
        if (key==ord('f') and controller) or (key==ord('d') and not(controller)):
           if mode=='user': 
            tRight(dts)
            Rcounter+=1
           else:
            if Rcounter<=Rcmax:
               tRight(0.75*dts)
               Rcounter+=1
        if (key==ord('m') and controller) or (key==32 and not(controller)): #space bar
           stop()

        Lcounter=max(Lcounter-1,0)
        Rcounter=max(Rcounter-1,0)
        #if (counter%2)==0: stop()

finally:

    # Stop streaming
    pipeline.stop()

print('reseting all servos...')
kit.servo[cam1].angle = def_conf[cam1]
kit.servo[cam2].angle = def_conf[cam2]

kit.servo[arm1].angle = nav_pos[arm1]
kit.servo[arm2].angle = nav_pos[arm2]
kit.servo[arm3].angle = nav_pos[arm3]
kit.servo[arm4].angle = nav_pos[arm4]
print('reseting all motors...')
reset()
