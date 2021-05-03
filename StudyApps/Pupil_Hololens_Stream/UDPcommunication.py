import socket
import threading
import numpy as np
from queue import Queue


class UDP_listener(threading.Thread):
    def __del__(self):
        if threading.Thread.is_alive(self):
            self.join()
        print("UDP Listener thread dead")

    def __init__(self, args, q, name='UDP Listener'):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.name = name
        self.args = args
        self.q = q
        self.receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_address = ('0,0,0,0', 3000)

    def Set_port(self, port):
        self.recv_address = ('0,0,0,0', port)

    def run(self):
        print('thread', threading.current_thread().getName(), ' started')
        self.receive_sock.bind(self.recv_address)
        self.receive_sock.settimeout(1)
        while self.args[0]:  # reads msg from other UDP senders
            try:
                msg, add = self.receive_sock.recvfrom(32)
                self.decode_message(msg)
                # evt = threading.Event()
                # self.q.put((msg, evt))
                # evt.wait()
            except Exception as e:
                print('error in receiving udp message', e)

    def decode_message(self, msg):
        print(msg)


class UDP_sender(threading.Thread):
    def __del__(self):
        if threading.Thread.is_alive(self):
            self.join()
        print("UDP Sender thread dead")

    def __init__(self, args, q, name='UDP Sender'):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.name = name
        self.args = args
        self.q = q
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.remote_ip = "192.168.0.9"
        self.destination = (self.remote_ip, 5005)

    def Set_destination(self, ip, port):
        self.remote_ip = ip
        self.destination = (self.remote_ip, port)

    def run(self) -> None:
        print('thread', threading.current_thread().getName(), ' started')
        while True:
            data, evt = self.q.get()
            if data is None:
                continue
            else:
                confidence = data['confidence']
                x = data['phi']
                y = data['theta']
                # print(confidence, x, y)
                self.send_message(confidence, x, y)
                evt.set()
                self.q.task_done()

    def send_message(self, confidence, x, y):
        try:
            Uint32_msg = self.encode_message(confidence, x, y)
            self.send_sock.sendto(Uint32_msg, self.destination)
        except Exception as e:
            print('error in sending udp message :', e)

    def encode_message(self, confidence, x, y):

        prefix = 1
        confidence = int(confidence * 7)

        x = int(x * 100)
        y = int(y * 100)
        x_sign = 0 if x >= 0 else 1
        y_sign = 0 if y >= 0 else 1
        x = abs(x)
        y = abs(y)
        if x > 8000:
            x = 8000
        if y > 8000:
            y = 8000

        # Actual message with 32-bit array
        send_binary = np.binary_repr(prefix, width=1) \
                      + np.binary_repr(confidence, width=3) \
                      + np.binary_repr(x_sign, width=1) \
                      + np.binary_repr(x, width=13) \
                      + np.binary_repr(y_sign, width=1) \
                      + np.binary_repr(y, width=13)
        # Unsigned 32 bit int ( UInt32 on C#)
        send_uint32 = (int(send_binary, 2)).to_bytes(4, 'big', signed=False)
        return send_uint32
