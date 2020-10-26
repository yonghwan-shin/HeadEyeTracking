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

# Initial Pupil-remote variables
ctx = zmq.Context()
pupil_remote = ctx.socket(zmq.REQ)
ip = "localhost"
port = 50020

ZMQ_stop = False


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
            return serial.Serial(port.device, 115200, write_timeout=0)


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
        subscriber.subscribe("pupil.1.3d")
    except KeyboardInterrupt:
        pass
    print("ZMQ start receiving")
    return subscriber


class ZMQ_listener(threading.Thread):
    def __del__(self):

        if threading.Thread.is_alive(self):
            self.join()
        print("ZMQ thread dead")

    def __init__(self, args, name="Pupil Listener"):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        # freq, mincutoff = 1.0, beta = 0.0, dcutoff = 1.0):
        # self.x_filter = OneEuroFilter(t0=time.time(),x0=0.0,min_cutoff=1,beta=0.1)
        # self.y_filter = OneEuroFilter(t0=time.time(), x0=0.0, min_cutoff=1, beta=0.1)
        self.DATA_ROOT = ""
        self.sub_num = 0
        self.filename = ""
        self.recording = False
        self.string2send = ["NaN", "NaN", "NaN", "NaN", "NaN"]
        self.name = name
        self.args = args
        self.stored_data = []

        self.buffer = ""
        self.timestapmes = []
        self.count = 1000
        self.sent = 1000
        self.Holo = find_holo_serial_port("/dev/cu.Bluetooth")
        self.Holo.flush()
        self.send_time = 0
        self.delays = []
        self.test_count = 0
        self.DATA = []
        self.sending = []
        self.frame_count = 0
        self.start_time = 0
        self.pupil_time = 0

    def run(self):
        print(threading.currentThread().getName(), "is started")
        subscriber = ZMQ_connect()
        sleep(1)  # wait a second before hearing pupil-data
        self.start_time = time.time()
        import msgpack

        while self.args[0]:
            try:

                topic, payload = subscriber.recv_multipart()

                message = msgpack.unpackb(
                    payload)  # undefined error (might be msgpack version conflict) , originally -> message = msgpack.unpackb(payload, encoding='utf-8')

                self.frame_count += 1
                if self.frame_count == 1:
                    self.pupil_time = message['timestamp']
                if self.Holo == None:
                    print("re-finding hololens port")
                    self.Holo = find_holo_serial_port("/dev/cu.Bluetooth")
                    continue
                """
                RECEIVING PART
                """
                self.buffer += self.Holo.read(self.Holo.in_waiting).decode()
                while "\n" in self.buffer:
                    data, self.buffer = self.buffer.split("\n", 1)
                    print('received :', data)
                    if data == "START":
                        self.recording = True
                    if data == "END":
                        # SAVE TEST
                        self.recording = False
                        print('finishing')
                        file = pd.DataFrame(self.DATA)
                        file.to_csv("log.csv", index=False)
                        sleep(1)

                    if data == "#" + str(self.sent) + "#" * 10:
                        self.delays.append((float(time.time()) - float(self.send_time)) * 1000)
                        print(
                            'chat delay',
                            format((float(time.time()) - float(self.send_time)) * 1000, '.3f'),
                            'ms',
                            'mean delay : ',
                            sum(self.delays) / len(self.delays)
                        )

                # norm_pos data form : [0.3894290751263883, 0.11579204756622086]

                """
                SENDING PART
                """
                # message length: prefix 1 + confidence 1 + delimiter 1 + postfix 1 + data 6*2 = 16
                prefix = 1
                confidence = int(message["confidence"] * 5)
                x = int(message["norm_pos"][0] * 10 ** 4)
                y = int(message["norm_pos"][1] * 10 ** 4)
                # Actual message with 32-bit array
                send_binary = np.binary_repr(prefix, width=1) \
                              + np.binary_repr(confidence, width=3) \
                              + np.binary_repr(x, width=14) \
                              + np.binary_repr(y, width=14)
                # Unsigned 32 bit int ( UInt32 on C#)
                send_uint32 = (int(send_binary, 2)).to_bytes(4, 'big', signed=False)

                self.Holo.write(send_uint32)

                if self.frame_count % 120 == 1:
                    pass

                if self.recording == True:
                    self.DATA.append(dict(
                        # timestamp=now,
                        # x=filtered_x,
                        # y=filtered_y,
                        # message = send_message
                    ))

            except KeyboardInterrupt:
                break

        sleep(0.1)
        self.join()
        return

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
