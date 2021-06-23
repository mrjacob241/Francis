import json

data={}

cmd='LEFT'
data[cmd]={'QUERY':['left'],'ANS':['turning!']}

cmd='RIGHT'
data[cmd]={'QUERY':['right'],'ANS':['turning!']}

cmd='FWD'
data[cmd]={'QUERY':['go forward','forward','4ward','fulwood','photo of'],'ANS':['on my way!']}

cmd='BACK'
data[cmd]={'QUERY':['go back','back'],'ANS':['on my way!']}

cmd='2xFWD'
data[cmd]={'QUERY':['go double forward','double forward'],'ANS':['on my way!']}

cmd='2xBACK'
data[cmd]={'QUERY':['go double back','double back'],'ANS':['on my way!']}

cmd='HLEFT'
data[cmd]={'QUERY':['slow left'],'ANS':['turning!']}
                    
cmd='HRIGHT'
data[cmd]={'QUERY':['slow right'],'ANS':['turning!']}

cmd='2xLEFT'
data[cmd]={'QUERY':['double left'],'ANS':['turning!']}
            
cmd='2xRIGHT'
data[cmd]={'QUERY':['double right'],'ANS':['turning!']}

cmd='HFWD'
data[cmd]={'QUERY':['go forward slow','slow forward'],'ANS':['on my way!']}

cmd='HBACK'
data[cmd]={'QUERY':['go back slow','slow back'],'ANS':['on my way!']}

cmd='GET_CLOSE'
data[cmd]={'QUERY':['get close'],'ANS':['ok!']}

cmd='GET_AWAY'
data[cmd]={'QUERY':['get away'],'ANS':['no problem!']}

cmd='QUIT'
data[cmd]={'QUERY':['quit'],'ANS':['see you soon!']}

cmd='AUTO'
data[cmd]={'QUERY':['auto','auto mode','follow me'],'ANS':['auto mode activated!']}

cmd='USER'
data[cmd]={'QUERY':['user','user mode','stop'],'ANS':['you have the control!']}

cmd='LOOK_LEFT'
data[cmd]={'QUERY':['look left'],'ANS':['ok!']}

cmd='LOOK_RIGHT'
data[cmd]={'QUERY':['look right'],'ANS':['ok!']}

cmd='LOOK_DOWN'
data[cmd]={'QUERY':['look down'],'ANS':['ok!']}

cmd='LOOK_FRONT'
data[cmd]={'QUERY':['look front','look forward'],'ANS':['ok!']}

cmd='LOOK_UP'
data[cmd]={'QUERY':['look up'],'ANS':['ok!']}
            
#miscellanea
cmd='HELLO'
data[cmd]={'QUERY':['hello boy','hello'],'ANS':['hi human!']}

cmd='BYE'
data[cmd]={'QUERY':['goodbye','bye'],'ANS':['see you soon!']}

cmd='TRACK_ME'
data[cmd]={'QUERY':['track me','track human','trek human'],'ANS':['tracking human!']}

cmd='TRACK_BALL'
data[cmd]={'QUERY':['trackball','track ball','truck ball','track red ball','track red bull','truck red bull'],'ANS':['tracking ball!']}

cmd='GRAB_BALL'
data[cmd]={'QUERY':['grab position','grab'],'ANS':['grabbing!']}

cmd='GET_BALL'
data[cmd]={'QUERY':['get red ball','get ball','get red bull','get bowl','get mall','get bull'],'ANS':['on my way!']}

cmd='FIND_BALL'
data[cmd]={'QUERY':['find ball','search ball','search bald','search bowl','search bold'],'ANS':['looking for the ball!']}

cmd='RELAX'
data[cmd]={'QUERY':['relax position','relax'],'ANS':['Fantastic!']}

cmd='CLOSE_HAND'
data[cmd]={'QUERY':['close hand','close'],'ANS':['closing hand!']}

cmd='LEAVE'
data[cmd]={'QUERY':['leave','live'],'ANS':['leaving!']}

cmd='LAUNCH_BALL'
data[cmd]={'QUERY':['launch','launch the ball','launch ball'],'ANS':['get the ball!']}

cmd='PLAY_SAMBA'
data[cmd]={'QUERY':["let's dance",'samba time','samba please'],'ANS':['samba time']}

cmd='STOP_SAMBA'
data[cmd]={'QUERY':["stop music",'stop samba','stop samba please','stop music please'],'ANS':['ok, it was funny!']}

with open('CMDS.json', 'w') as outfile:
    json.dump(data, outfile, indent=3)