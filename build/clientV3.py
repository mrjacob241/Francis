import time, socket, sys
import androidhelper

droid = androidhelper.Android()
# modify language from phone settings 
#in speech synthesis and system language

def speak(text):
    droid.ttsSpeak(text)
    while droid.ttsIsSpeaking()[1] == True:
        time.sleep(1)

def listen():
    return droid.recognizeSpeech('Speak Now','en',None)[1]

 
socket_server = socket.socket()
server_host = socket.gethostname()
ip = socket.gethostbyname(server_host)
#ip="192.168.1.65"
sport = 8063
 
print('This is your IP address: ',ip)
#server_host = input("Enter friend's IP address: ")
name = input("Enter Friend's name: ")

#server_host = "192.168.1.68"
server_host = "192.168.1.80"
#server_host = "192.168.43.60"

 
socket_server.connect((server_host, sport))
 
socket_server.send(name.encode())
server_name = socket_server.recv(1024)
server_name = server_name.decode()
 
print(server_name,' has joined...')
speak('connection established!')
while True:
    message = (socket_server.recv(1024)).decode()
    print(server_name, ":", message)
    input("press enter...")
    message = listen()
    if message=='exit':
        break
    socket_server.send(message.encode())
    print("Me :", message)
    reply = (socket_server.recv(1024)).decode()
    if reply[0:4].lower()=='msg:':
        print(server_name, ":", reply[4:])
        speak(reply[4:])
    else:
        print(server_name, ":", reply)
