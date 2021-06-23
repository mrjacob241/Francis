from multiprocessing import Process
from playsound import playsound
import os


def kill(p):
    while not(os.path.exists('queque/samba_alt.txt')):
       print('player active')
    p.terminate()
    os.remove('queque/samba_alt.txt')

def playsong(title):
    playsound(title)
    fc = open('queque/samba_alt.txt','w+')
    fc.write('terminate')
    fc.close()
    fc = open('queque/samba_stop.txt','w+')
    fc.write('terminate')
    fc.close()


if __name__ == '__main__':
    while True:
       if os.path.exists('queque/samba_start.txt'):
          fs = open('queque/samba_start.txt','r')
          title=fs.read()
          fs.close()
          if os.path.exists('queque/samba_alt.txt'):
              os.remove('queque/samba_alt.txt')
          if os.path.exists('queque/samba_stop.txt'):
              os.remove('queque/samba_stop.txt')
          p = Process(target=playsong, args=(title,))
          p.start()
          k = Process(target=kill, args=(p,))
          k.start()
          os.remove('queque/samba_start.txt')

