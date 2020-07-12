import time
from time import sleep
import serial
import serial.tools.list_ports
import threading

import os
import csv
import math

connected = False

arduino = serial.Serial()

dataline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

IMU_DATA_ROOT = ""
IMU_SUBJECT = 0
IMU_CURRFILE = ""
IMU_RECORDING = ""


def connectArduino():
    ports = serial.tools.list_ports.comports()
    print('reconnecting')
    for port in ports:
        # print(port.device)
        if port.device.startswith('/dev/cu.usb'):
            return serial.Serial(port.device, 9600);


def read_from_arduino(port):
    while True:
        try:
            if port == None:
                port = connectArduino()

            res = port.readline()
            global dataline
            dataline = res.decode()[:len(res) - 3].split(',')
            if IMU_RECORDING == "RECORD":
                savefile_arduino(dataline)


        except:
            if not port == None:
                port.close()
                port = None
                print('disconnecting')
                dataline = [0.0, 0.0, 0.0, 0.0, 0.0]
            time.sleep(2)


def savefile_arduino(data, _filename):
    if os.path.isfile(
            IMU_DATA_ROOT + "/" + IMU_SUBJECT + "/" + "IMU_" + IMU_CURRFILE + ".csv"):  # Path location 할당해줘야합니다. 초기 헤더값 이후 데이터 어펜드.
        f = open(IMU_DATA_ROOT + "/" + IMU_SUBJECT + "/" + "IMU_" + IMU_CURRFILE + ".csv", 'a')
        wr = csv.writer(f, lineterminator='\n')
        # stop = timeit.default_timer()
        ts = time.time()
        current_time = []
        current_time.append(ts)
        wr.writerow(data + current_time)
    else:  # 초기 헤더값 설정
        f = open(IMU_DATA_ROOT + "/" + IMU_SUBJECT + "/" + "IMU_" + IMU_CURRFILE + ".csv", 'w')
        wr = csv.writer(f, lineterminator='\n')
        # trial_start_time = timeit.default_timer()
        wr.writerow(["quatI", "quatJ", "quatK", "quatReal", "quatRadianAccuracy", "IMUtimestamp"])


class IMU_listener(threading.Thread):

    def __del__(self):
        if threading.Thread.is_alive(self):
            self.join()
        print("Holo thread dead")

    def __init__(self, args, name="IMU Listener"):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True

        self.name = name
        self.args = args
        self.DATA_ROOT = ""
        self.sub_num = 0
        self.filename = ""
        self.recording = False
        self.stored_data = []
        self.buffer = ''
        self.timer = 0

    def run(self):
        print(threading.current_thread().getName(), 'is started')
        IMU = connectArduino()
        sleep(1)
        while self.args[0]:
            try:
                if IMU == None:
                    IMU = connectArduino()

                # while IMU.in_waiting >0:
                self.buffer += IMU.read(IMU.in_waiting).decode()
                while '\n' in self.buffer:
                    data, self.buffer = self.buffer.split('\n', 1)
                    now = time.time()
                    if self.recording:
                        dataline = data[:-3].split(',')
                        self.stored_data.append([str(now)] + dataline)
                    else:
                        if (now - math.floor(now)) < 0.003:
                            print('IMU: ', data)

            except:
                if not IMU == None:  # Handle disconnection error
                    IMU.close()
                    IMU = None
                    print('disconnected')
                    dataline = [0.0, 0.0, 0.0, 0.0, 0.0]
            # print('something other error')
            # time.sleep(0.5)

    def save_data(self):
        full_name = 'IMU_' + self.filename + '.csv'
        file_path = os.path.join(self.DATA_ROOT, str(self.sub_num), full_name)

        # if os.path.isfile(file_path):
        # 	f = open(file_path,'a')
        # else:
        f = open(file_path, 'w')
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['imu_packets', len(self.stored_data)])
        wr.writerow(["IMUtimestamp", "quatI", "quatJ", "quatK", "quatReal", "quatRadianAccuracy"])
        for line in self.stored_data:
            # print(line)
            wr.writerow(line)
        f.close()
        print('saved', full_name, ' total', len(self.stored_data), 'imu packets')
        self.stored_data.clear()

    def End_trial(self):
        self.recording = False
        self.save_data()

    # print('trial takes' , time.time() - self.timer, 'seconds')
    # self.stored_data.clear()	#just to be sure
    def Start_trial(self):
        # self.stored_data.clear()	#just to be sure
        self.recording = True
        self.timer = time.time()

    def Set_filename(self, _filename):
        self.filename = _filename

    def Set_sub_num(self, _sub_num):
        self.sub_num = _sub_num


def main():
    imu_receiver = IMU_listener(name='imu listener', args=[True])
    imu_receiver.start()


if __name__ == "__main__":
    main()
