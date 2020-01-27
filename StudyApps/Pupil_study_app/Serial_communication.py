import time
import threading
from time import sleep
import serial
import serial.tools.list_ports

print('serial ', serial.__version__)

# set a port number & baud rate
list = serial.tools.list_ports.comports()
arduinoPORT = ""
hololensPORT = ""
arduinoBaudRate = 115200
hololensBaudRate = 115200

arduinoPORT = '/dev/cu.usbmodem14101'


# print(list)
class Serial_Listener(threading.Thread):
    def __del__(self):
        if threading.Thread.isAlive(self):
            self.join()
        print("Serial communication dead")

    def __init__(self, args, name="Serial comm - Listener"):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.name = name
        self.args = args
        # try:
        # self.arduino, self.hololens = connect_serial()
        # except TypeError:
        #     print("no port")

    def run(self):
        print(threading.currentThread().getName(), " is started")

        while self.args[0]:
            try:
                serial_listening()
            except KeyboardInterrupt:
                break
        sleep(0.1)
        self.join()
        print(threading.currentThread().getName(), ' end')
        return


class Serial_Sender(threading.Thread):
    def __del__(self):
        if threading.Thread.isAlive(self):
            self.join()
        print("Serial communication dead")

    def __init__(self, args, name="Serial comm - Sender"):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.name = name
        self.args = args

    def run(self):
        print(threading.currentThread().getName(), " is started")

        while self.args[0]:
            try:
                getHololens()

            except KeyboardInterrupt:
                break

        sleep(0.1)
        self.join()
        print('end')
        return


for element in list:
    print("PORT:", str(element.device))

for port, desc, hwid in sorted(list):
    print(f"{port}: {desc} {hwid}")
    if "Bl" in port:
        hololensPORT = port


def connect_serial():
    arduino = serial.Serial(arduinoPORT, arduinoBaudRate)
    hololens = serial.Serial(hololensPORT, hololensBaudRate)

    return arduino, hololens


def serial_listening():
    arduino, hololens = connect_serial()
    while True:
        if arduino.readable():
            Quat = arduino.readline()
        #     do something?
        if hololens.readable():
            signal_hololens = hololens.readline()
            #     do something?


def getQuat():
    arduino = serial.Serial(arduinoPORT, arduinoBaudRate)
    while True:
        if arduino.readable():
            res = arduino.readline()


def getHololens():
    hololens = serial.Serial(hololensPORT, hololensBaudRate)
    while True:
        if hololens.readable():
            res = hololens.readline()


def sendtoHololens(string2send):
    hololens = serial.Serial(hololensPORT, hololensBaudRate)
    hololens.write(string2send.encode("UTF-8"))
#
# while(True):
# 	info = "#0.1$0.9$0.9$@"
# 	# Serial.write(info)
# 	Serial.write(info.encode('UTF-8'))
# 	time.sleep(0.05)
# 	# info = "#0.9$0.1$0.9$"
# 	# # Serial.write(info)
# 	# Serial.write(info.encode('UTF-8'))
# 	# time.sleep(0.05)
# 	print(Serial.read(Serial.inWaiting()))

# if __name__ == "__main__":
