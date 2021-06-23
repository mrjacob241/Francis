import time, socket, sys, json
 
new_socket = socket.socket()
host_name = socket.gethostname()
s_ip = socket.gethostbyname(host_name)

with open('CMDS.json') as json_file:
    CMDS = json.load(json_file)
CMDS_LIST = CMDS.keys()
print('commands loaded!')
 
port = 8063
 
new_socket.bind((host_name, port))
print( "Binding successful!")
print("This is your IP: ", s_ip)
 
name = input('Enter name: ')
 
new_socket.listen(1) 
 
 
conn, add = new_socket.accept()
 
print("Received connection from ", add[0])
print('Connection Established. Connected From: ',add[0])
 
client = (conn.recv(1024)).decode()
print(client + ' has connected.')
 
conn.send(name.encode())
while True:
    try:
    #if True:
        #message = input('Me : ')
        message = "waiting query..."
        conn.send(message.encode())
        message = conn.recv(1024)
        message = message.decode()
        if len(message)>0:
            print(client, ':', message)
            f=open('queque_number.txt','r')
            qn = int(f.read())
            print('queque:',qn)
            f.close()
            f=open('queque/'+str(qn+1)+'.txt','w+')
            f.write(message)
            f.close()
        flag = False
        reply = 'None'
        for cmd in CMDS_LIST:
            if message in CMDS[cmd]['QUERY']:
               flag = True
               reply = 'MSG:'+CMDS[cmd]['ANS'][0]
        if message!='exit':
            conn.send(reply.encode())

    except:
        print('connection lost!')
        break

