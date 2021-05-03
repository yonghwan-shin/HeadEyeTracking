import threading
import time
from time import sleep
import zmq
import serial
import serial.tools.list_ports
import os
import numpy as np
import csv
import pandas as pd
import socket
import math

# Initial Pupil-remote variables
from Naming import *

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

ctx = zmq.Context()
pupil_remote = ctx.socket(zmq.REQ)
ip = "localhost"
port = 50020

ZMQ_stop = False


def checkDirectory(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def find_holo_serial_port(portname: str):
    """
    Find the desired serial port and connect within available serial ports
    :str portname: Desired port's name
    :return: pyserial object with baudrate 115200
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.device.startswith(portname):
            print('find the port! : ', port.device)
            return serial.Serial(port.device, 115200, write_timeout=0, timeout=0)


def ZMQ_connect():
    """
    Connect to the Pupil-remote (specifically, pupil.1.3d which is right eye cam)
    :return: ZMQ subscriber for right eye
    """
    try:
        pupil_remote.connect(f"tcp://{ip}:{port}")
        # Request 'SUB_PORT' for reading data
        pupil_remote.send_string("SUB_PORT")
        sub_port = pupil_remote.recv_string()

        # Request 'PUB_PORT' for writing data
        pupil_remote.send_string("PUB_PORT")
        pub_port = pupil_remote.recv_string()

        subscriber = ctx.socket(zmq.SUB)
        subscriber.connect(f"tcp://{ip}:{sub_port}")
        subscriber.subscribe("pupil.0.3d")
    except KeyboardInterrupt:
        pass
    print("ZMQ start receiving")
    return subscriber


class ZMQ_listener(threading.Thread):
    def __del__(self):

        if threading.Thread.is_alive(self):
            self.join()
        print("ZMQ thread dead")

    def __init__(self, args, q, name="Pupil Listener"):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        # freq, mincutoff = 1.0, beta = 0.0, dcutoff = 1.0):
        # self.x_filter = OneEuroFilter(t0=time.time(),x0=0.0,min_cutoff=1,beta=0.1)
        # self.y_filter = OneEuroFilter(t0=time.time(), x0=0.0, min_cutoff=1, beta=0.1)
        self.DATA_ROOT = DATA_ROOT
        self.sub_num = 0
        self.filename = ""
        self.recording = False

        self.name = name
        self.args = args
        self.q = q
        self.stored_data = []
        self.holodata = ["#START"]
        self.filenames = []
        self.curr_file = ""
        self.buffer = ""
        self.timer = 0
        self.delay_timer = 0
        self.median_filter_window = 181
        self.median_phi_list = []
        self.median_theta_list = []
        self.phi1 = 0
        self.phi2 = 0
        self.theta1 = 0
        self.theta2 = 0
        self.Holo = find_holo_serial_port("/dev/cu.Bluetooth")
        self.online = True

    def dejitter_single(self, second_previous, previous, new):
        p = previous
        if (previous < new and previous < second_previous) or (second_previous < previous and new < previous):
            if abs(second_previous - new) < abs(new - previous):
                p = second_previous
            else:
                p = new
        return p, new

    def dejitter_double(self, x3, x2, x1, x):
        g1 = x1
        g2 = x2
        if (x2 == x1) and not (x == x1 or x2 == x3):
            if abs(x1 - x) < abs(x1 - x3):
                g1 = x
                g2 = x
            else:
                g1 = x3
                g2 = x3
        return g2, g1, x

    def run(self):
        print(threading.currentThread().getName(), "is started")
        checkDirectory(DATA_ROOT)
        self.sub_singal()
        self.Holo.flush()
        subscriber = ZMQ_connect()
        sleep(1)  # wait a second before hearing pupil-data

        import msgpack
        # ip = socket.gethostbyname_ex(socket.gethostname())[-1][-1]
        # print(ip)
        # remote_ip = "192.168.0.9"
        # send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # destination = (remote_ip, 5005)
        # receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # recv_address = ('0.0.0.0', 3000)
        # receive_sock.bind(recv_address)
        # receive_sock.settimeout(3)
        data_size = 5000
        cnt = 0
        t = time.time()
        while self.args[0]:
            try:

                topic, payload = subscriber.recv_multipart()

                message = msgpack.unpackb(payload)
                pos = message['norm_pos']
                confidence = message['confidence']
                phi = message['phi']
                theta = message['theta']

                # median filtering
                if len(self.median_phi_list) >= self.median_filter_window:
                    self.median_phi_list.pop(0)
                if len(self.median_theta_list) >= self.median_filter_window:
                    self.median_theta_list.pop(0)
                if confidence > 0.6:
                    self.median_phi_list.append(phi)
                    self.median_theta_list.append(theta)
                else:
                    self.median_phi_list.append(None)
                    self.median_theta_list.append(None)
                try:
                    median_phi = np.median(list(filter(None, self.median_phi_list)))
                    median_theta = np.median(list(filter(None, self.median_theta_list)))
                except:
                    median_phi = 0
                    median_theta = 0
                # print(median_theta)
                phi -= median_phi
                theta -= median_theta
                phi = phi * 180 / math.pi
                theta = theta * 180 / math.pi
                self.phi1 = self.single_item_jitter(self.phi2, self.phi1, phi)
                self.theta1 = self.single_item_jitter(self.theta2, self.theta1, theta)

                eye_msg = {'confidence': confidence, 'phi': self.phi1, 'theta': self.theta1}
                if self.online:
                    evt = threading.Event()
                    self.q.put((eye_msg, evt))
                    evt.wait()
                # print(phi,theta)
                # conf_str = "{:.1f}".format(confidence)
                # phi_str = "{:.2f}".format(phi)
                # theta_str = "{:.2f}".format(theta)

                # message length: prefix 1 + confidence 1 + delimiter 1 + postfix 1 + data 6*2 = 16
                # prefix = 1
                # confidence = int(confidence * 7)
                # x = int(phi * 10 ** 2)
                # y = int(theta * 10 ** 2)
                #
                # x_sign = 0 if x >= 0 else 1
                # y_sign = 0 if y >= 0 else 1
                # x = abs(x)
                # y = abs(y)
                #
                # # Actual message with 32-bit array
                # send_binary = np.binary_repr(prefix, width=1) \
                #               + np.binary_repr(confidence, width=3) \
                #               + np.binary_repr(x_sign, width=1) \
                #               + np.binary_repr(x, width=13) \
                #               + np.binary_repr(y_sign, width=1) \
                #               + np.binary_repr(y, width=13)
                # # Unsigned 32 bit int ( UInt32 on C#)
                # send_uint32 = (int(send_binary, 2)).to_bytes(4, 'big', signed=False)

                # msg = bytes(conf_str+","+phi_str + "," + theta_str, 'utf-8')

                # send_time = time.time()
                # send_sock.sendto(send_uint32, destination)
                # undefined error (might be msgpack version conflict) , originally -> message = msgpack.unpackb(
                # payload, encoding='utf-8')

                # if self.Holo == None:
                #     print("re-finding hololens port")
                #     self.Holo = find_holo_serial_port("/dev/cu.Bluetooth")
                #     self.Holo.flush()
                #     continue
                """
                RECEIVING PART
                """
                # self.buffer += self.Holo.read(self.Holo.in_waiting).decode()
                # while "\n" in self.buffer:
                #     data, self.buffer = self.buffer.split("\n", 1)
                #     if data.startswith("#"):
                #         self.holodata.append(data)
                #     print('received :', data)
                # self.Holo_START()
                # self.Holo_INIT()
                # # TRIAL -> file save
                # self.Holo_TRIAL()
                # # END -> NEXT or BREAK or FINISH, NEXT: INIT 상태로 만들기.
                # self.Holo_END()

                # norm_pos data form : [0.3894290751263883, 0.11579204756622086]

                """
                SENDING PART
                """
                # message length: prefix 1 + confidence 1 + delimiter 1 + postfix 1 + data 6*2 = 16
                # prefix = 1
                # confidence = int(message["confidence"] * 7)
                # x = int(message["norm_pos"][0] * 10 ** 4)
                # y = int(message["norm_pos"][1] * 10 ** 4)
                # # Actual message with 32-bit array
                # send_binary = np.binary_repr(prefix, width=1) \
                #               + np.binary_repr(confidence, width=3) \
                #               + np.binary_repr(x, width=14) \
                #               + np.binary_repr(y, width=14)
                # # Unsigned 32 bit int ( UInt32 on C#)
                # send_uint32 = (int(send_binary, 2)).to_bytes(4, 'big', signed=False)

                # self.Holo.write(send_uint32)
                self.phi2 = self.phi1
                self.phi1 = phi
                self.theta2 = self.theta1
                self.theta1 = theta

                if self.recording:
                    self.stored_data.append([str(time.time()), str(message)])

            # except KeyboardInterrupt:
            except Exception as e:
                print('error in reading pupil-labs', e)
                # break

        sleep(0.1)
        self.join()
        return

    def single_item_jitter(self, x2, x1, x):
        g = x1
        if (x1 < x and x1 < x2) or (x2 < x1 and x < x1):
            if abs(x2 - x) < abs(x - x1):
                g = x2
            else:
                g = x
        x1 = g
        return x1

    def double_item_jitter(self, x3, x2, x1, x):
        g1 = x1
        g2 = x2
        if (x2 == x1) and not (x == x1 or x2 == x3):
            if abs(x1 - x) < abs(x1 - x3):
                g1 = x
                g2 = x
            else:
                g1 = x3
                g2 = x3
        x1 = g1
        x2 = g2
        return x2, x1

    def send_to_hololens(self, msg: str):
        """
        Send String message to serial port
        :param msg: message string
        :return: None
        """
        self.Holo.write(msg.encode("UTF-8"))

    def send_bytes(self, msg: bytes):
        """
        Send bytes (bit array) to serial port
        :param msg: message bytes
        :return: None
        """
        self.Holo.write(msg)

    def send_subject_to_hololens(self):
        prefix = 0
        binary = np.binary_repr(prefix, width=1) + np.binary_repr(0, width=2) + \
                 np.binary_repr(0, width=3) + np.binary_repr(0, width=1) + \
                 np.binary_repr(0, width=3) + np.binary_repr(0, width=3) + \
                 np.binary_repr(int(self.sub_num), width=10) + np.binary_repr(0, width=9)
        send_uint32 = (int(binary, 2)).to_bytes(4, 'big', signed=False)
        self.Holo.write(send_uint32)

    def send_next_trial_to_hololens(self, _filename):
        prefix = 0
        target, environment, block, C, subnum = decompose_walkonly_file_name(_filename)
        env = 0 if environment == 'U' else 1
        binary = np.binary_repr(prefix, width=1) + np.binary_repr(1, width=2) + \
                 np.binary_repr(int(target), width=3) + np.binary_repr(env, width=1) + \
                 np.binary_repr(int(block), width=3) + np.binary_repr(int(C), width=3) + \
                 np.binary_repr(int(subnum), width=10) + np.binary_repr(0, width=9)
        send_uint32 = (int(binary, 2)).to_bytes(4, 'big', signed=False)
        self.Holo.write(send_uint32)

    def send_break_to_hololens(self):
        prefix = 0
        binary = np.binary_repr(prefix, width=1) + np.binary_repr(2, width=2) + \
                 np.binary_repr(0, width=32 - 3)
        send_uint32 = (int(binary, 2)).to_bytes(4, 'big', signed=False)
        self.Holo.write(send_uint32)

    def send_finish_to_hololens(self):
        prefix = 0
        binary = np.binary_repr(prefix, width=1) + np.binary_repr(3, width=2) + \
                 np.binary_repr(0, width=32 - 3)
        send_uint32 = (int(binary, 2)).to_bytes(4, 'big', signed=False)
        self.Holo.write(send_uint32)

    def Holo_START(self):
        if self.holodata[-1] == "#START":
            self.send_subject_to_hololens()
            print('sending sub', self.sub_num)
            # self.send_to_hololens("#SUB" + str(self.sub_num))
            sleep(2)

    def Holo_INIT(self):
        if (self.holodata[-1] == "#INIT") and (self.holodata[-2] != "#INIT"):
            curr_file = current_add(self.filenames.pop())
            self.Set_filename(curr_file)

            print('remaining trials:', len(self.filenames))
            self.send_next_trial_to_hololens(curr_file)
            # self.send_to_hololens("#NEXT_" + curr_file)
            self.holodata.append("#INIT")
            self.delay_timer = time.time()

    def Holo_TRIAL(self):
        if (self.holodata[-1] == "#TRIAL") and (self.holodata[-2] != "#TRIAL"):
            # imu_thread.Start_trial()
            # zmq_thread.Start_trial()
            # print('start delay :', time.time()-self.delay_timer)

            self.Start_trial()
            self.timer = time.time()
            # global timer
            # timer = time.time()
            self.holodata.append("#TRIAL")

    def Holo_END(self):
        if (self.holodata[-1] == "#END"):
            # global timer
            # global now
            # imu_thread.End_trial()
            # zmq_thread.End_trial()
            self.End_trial()
            now = time.time()
            print('-' * 20, 'END', '-' * 20, end=' ')

            print('time:', "%.3f" % (now - self.timer), 'sec')
            if self.filenames[-1] == 'BREAK' and (self.holodata[-2] != "BREAK"):
                # self.send_to_hololens("#BREAK")
                self.send_break_to_hololens()
                self.filenames.pop()
                self.holodata.append('BREAK')
            elif self.filenames[-1] == 'FINISH' and (self.holodata[-2] != "FINISH"):
                # self.send_to_hololens("#FINISH")
                self.send_finish_to_hololens()
                self.filenames.pop()
                self.holodata.append('FINISH')
            else:
                self.holodata.append("#INIT")

    def sub_singal(self):  # init

        print('making filenames for subject ', self.sub_num)

        self.filenames = make_experiment_array_walkonly(self.sub_num)

        print('files :', self.filenames)

        # if restart:
        #     for _ in range(restart_number):
        #         filename.pop()
        checkDirectory(DATA_ROOT + "/" + str(self.sub_num))

    def save_data(self):
        full_name = "EYE_" + self.filename + ".csv"
        file_path = os.path.join(self.DATA_ROOT, str(self.sub_num), full_name)

        # if os.path.isfile(file_path):
        # 	f = open(file_path,'a')
        # else:
        f = open(file_path, "w")

        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["eye_packets", len(self.stored_data)])
        wr.writerow(["norm_X", "norm_Y", "phi", "theta", "confidence", "timestamp"])
        for line in self.stored_data:
            wr.writerow(line)
        f.close()
        print("saved", full_name, " total", len(self.stored_data), "eye packets")
        self.stored_data.clear()

    def End_trial(self):
        self.recording = False
        self.save_data()
        # self.stored_data.clear()	#just to be sure

    def Start_trial(self):
        # self.stored_data.clear()	#just to be sure
        self.recording = True

    def Set_filename(self, _filename):
        self.filename = _filename

    def Set_sub_num(self, _sub_num):
        print('Subject number set as : ', _sub_num)
        self.sub_num = _sub_num


def ZMQ_listener_main():
    zmq_receiver_name = "ZMQ_listener"
    zmq_thread = ZMQ_listener(name="ZMQ_listener", args=(zmq_receiver_name, True))
    zmq_thread.start()


def savefile_ZMQ(self, data):
    if os.path.isfile(
            self.ZMQ_DATA_ROOT
            + "/"
            + self.ZMQ_SUBJECT
            + "/"
            + "EYE_"
            + self.ZMQ_CURRFILE
            + ".csv"
    ):  # Path location 할당해줘야합니다. 초기 헤더값 이후 데이터 어펜드.
        f = open(
            self.ZMQ_DATA_ROOT
            + "/"
            + self.ZMQ_SUBJECT
            + "/"
            + "EYE_"
            + self.ZMQ_CURRFILE
            + ".csv",
            "a",
        )
        wr = csv.writer(f, lineterminator="\n")
        # stop = timeit.default_timer()
        ts = time.time()
        current_time = []
        current_time.append(ts)
        wr.writerow(data + current_time)
    else:  # 초기 헤더값 설정
        f = open(
            self.ZMQ_DATA_ROOT
            + "/"
            + self.ZMQ_SUBJECT
            + "/"
            + "EYE_"
            + self.ZMQ_CURRFILE
            + ".csv",
            "w",
        )
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(
            ["zmq_X", "zmq_Y", "phi", "theta", "zmq_confidence", "ZMQtimestamp"]
        )


if __name__ == "__main__":
    Pupil_thread = ZMQ_listener(name="Pupil Listener", args=[True])
    Pupil_thread.start()
    while True:
        pass
