import time, socket, sys

new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
s_ip = socket.gethostbyname(host_name)
#s_ip="192.168.1.233"

port = 8080

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
#new_socket.setblocking(False)
c=0
while True:
    #message = input('Me : ')
    print(c)
    #conn.send(message.encode())
    message = conn.recv(1024)
    message = message.decode()
    print(client, ':', message)
    with open('status.txt','w+') as f:
        f.write(message)
    
    c+=1
