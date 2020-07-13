import threading
import time
from time import sleep
import zmq
import msgpack
import os
import csv

ctx = zmq.Context()
pupil_remote = ctx.socket(zmq.REQ)
ip = 'localhost'
# ip = '192.168.0.49'
port = 50020
ZMQ_stop = False


def zmq_connect():
    try:
        pupil_remote.connect(f'tcp://{ip}:{port}')
        # Request 'SUB_PORT' for reading data
        pupil_remote.send_string('SUB_PORT')
        sub_port = pupil_remote.recv_string()

        # Request 'PUB_PORT' for writing data
        pupil_remote.send_string('PUB_PORT')
        pub_port = pupil_remote.recv_string()

        subscriber = ctx.socket(zmq.SUB)
        subscriber.connect(f"tcp://{ip}:{sub_port}")
        subscriber.subscribe('pupil.1')
        print('ZMQ start receiving')
        return subscriber

    except KeyboardInterrupt:
        pass


class PupilListener(threading.Thread):
    def __del__(self):
        if threading.Thread.is_alive(self):
            self.join()
        print("Thread : Pupil closed")

    def __init__(self):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.root = ''
        self.sub_num = 0
        self.filename = ''
        self.recording = False
        self.name = 'Pupil_Listener'
        self.stored_data = []
        self.active = True

    def run(self):
        print(threading.currentThread().getName(), 'activated')
        sleep(1)
        subscriber = zmq_connect()
        sleep(1)

        while self.active:
            try:
                topic, payload = subscriber.recv_multipart()
                message = msgpack.unpackb(payload, encoding='utf-8')
                if self.recording:
                    self.stored_data.append([str(time.time()), str(message)])
            except:
                break
            sleep(0.1)
            self.join()
            return

    def save_data(self):
        full_name = 'EYE_' + self.filename + '.csv'
        file_path = os.path.join(self.root, str(self.sub_num), full_name)
        f = open(file_path, 'w')
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['eye_packets', len(self.stored_data)])
        for line in self.stored_data:
            wr.writerow(line)
        f.close()
        print('saved', full_name, ' total', len(self.stored_data), 'eye packets')
        self.stored_data.clear()

    def end_trial(self):
        self.recording = False
        self.save_data()

    def start_trial(self):
        self.recording = True

    def set_filename(self, _filename):
        self.filename = _filename

    def set_sub_num(self, _sub_num):
        self.sub_num = _sub_num
