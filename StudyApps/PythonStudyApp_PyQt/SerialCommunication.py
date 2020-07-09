import serial
import serial.tools.list_ports
import time
import threading
import os
import csv


def connect_arduino():
    ports = serial.tools.list_ports.comports()
    print('connecting: arduino')
    for port in ports:
        if port.device.startswith('/dev/cu.usbmodem'):
            return serial.Serial(port.device, 9600);


def connect_hololens():
    ports = serial.tools.list_ports.comports()
    print('connecting: hololens')
    for port in ports:
        if port.device.startswith('/dev/cu.Bluetooth'):
            return serial.Serial(port.device, 115200);


def Holo_data_receive(buffer, Holo):
    buffer += Holo.read(Holo.in_waiting).decode()
    while '\n' in buffer:
        data, buffer = buffer.split('\n', 1)
        if data.startswith('#'):
            pass
            # holodata.append(data)
        print('received:', data)


class ImuListener(threading.Thread):

    def __del__(self):
        if threading.Thread.is_alive(self):
            self.join()
        print('Thread: IMU closed')

    def __init__(self):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.name = "IMU_Listener"
        self.root = ''
        self.active = True
        self.sub_num = 0
        self.buffer = ''
        self.stored_data = []
        self.filename = ''
        self.recording = False

    def run(self):
        print(threading.current_thread().getName(), 'activated')
        time.sleep(1)
        imu_port = connect_arduino()
        time.sleep(1)
        while self.active:
            try:
                if imu_port is None: time.sleep(0.5);imu_port = connect_arduino()
                self.buffer += imu_port.read(imu_port.in_waiting).decode()
                while '\n' in self.buffer:
                    data, self.buffer = self.buffer.split('\n', 1)
                    now = time.time()
                    if self.recording:
                        datalist = data[:-3].split(',')
                        self.stored_data.append([str(now)] + datalist)

            except:
                if imu_port is not None:
                    imu_port.close()
                    imu_port = None
                print('Disconnect : IMU')

    def save_trial(self):
        full_name = 'IMU_' + self.filename + '.csv'
        file_path = os.path.join(self.root, str(self.sub_num), full_name)
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

    def end_trial(self):
        self.recording = False
        self.save_trial()

    def start_trial(self):
        self.recording = True

    def set_filename(self, _filename):
        self.filename = _filename

    def set_subject_number(self, _sub_num):
        self.sub_num = _sub_num


class Hololens():
    def __init__(self):
        self.port = connect_hololens()
        self.buffer = []
        self.now = 0
        self.received = []

    def send(self, message):
        self.port.write(message.encode("UTF-8"))
        print("Send to Hololens : ", message.encode("UTF-8"))

    def read(self):
        if self.port.in_waiting > 0:
            self.buffer.append(self.port.read(self.port.in_waiting).decode('utf-8'))
            for component in self.buffer:
                if component.startswith('#'):
                    self.received.append(component)
                print(f'received: {component}')
            self.buffer.clear()
