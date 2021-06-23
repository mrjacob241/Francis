#test
#python3 /usr/local/lib/python3.8/dist-packages/evdev/evtest.py
#python3 /home/jacob/.local/lib/python3.6/site-packages/evdev/evtest.py

#import evdev
from evdev import InputDevice, categorize, ecodes

#creates object 'gamepad' to store the data
#you can call it whatever you like
gamepad = InputDevice('/dev/input/event21')

#button code variables (change to suit your device)
aBtn = 34
bBtn = 36
xBtn = 35
yBtn = 23

up = 46
down = 32
left = 18
right = 33

start = 24
select = 49

lTrig = 37
rTrig = 50

#prints out device info at start
print(gamepad)

#loop and filter by event code and print the mapped label
for event in gamepad.read_loop():
    if event.type == ecodes.EV_KEY:
        void=[]
        if event.value == 1:
            #print(event)
            if event.code == yBtn:
                print("Y")
            elif event.code == bBtn:
                print("B")
            elif event.code == aBtn:
                print("A")
            elif event.code == xBtn:
                print("X")

            elif event.code == up:
                print("up")
            elif event.code == down:
                print("down")
            elif event.code == left:
                print("left")
            elif event.code == right:
                print("right")

            elif event.code == start:
                print("start")
            elif event.code == select:
                print("select")

            elif event.code == lTrig:
                print("left bumper")
            elif event.code == rTrig:
                print("right bumper")
