import serial
import serial.tools.list_ports
import threading
import time
from time import sleep
	
def connectHolo():  # init
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.device.startswith('/dev/cu.Bluetooth'):
            # if port.device.startswith('/dev/cu.DESKTOP'):
            return serial.Serial(port.device, 115200)

class Hololens_Serial(threading.Thread):
    def __del__(self):
        if threading.Thread.is_alive(self):
            self.join()
        print(self.name,'closed')
    def __init__(self,name='Hololens Serial Port'):
        threading.Thread.__init__(self)
        threading.Thread.daemon=True

        self.name = name
        self.buffer=''
    def run(self):
        print(threading.current_thread().getName(), 'is started')
        Holo = connectHolo()
        sleep(1)
        while True:
            try:
                if Holo == None:
                    Holo = connectHolo()
                self.buffer += Holo.read(Holo.in_waiting).decode()
                while '\n' in self.buffer:
                    data,self.buffer = self.buffer.split('\n',1)
                    
            except Exception as e:
                print('Hololens error:', e.args)
                if not Holo == None:
                    Holo.close()
                    Holo = None
                    print('disconnected')

            