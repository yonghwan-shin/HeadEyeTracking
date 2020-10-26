from zmq_pupil import *
import socket


def bytes_tp_int(bytes):
    result = 0
    for b in bytes:
        result = result * 256 + int(b)
    return result


def int_to_bytes(value, length):
    result = []
    for i in range(0, length):
        result.append(value >> (i * 8) & 0xff)
    result.reverse()
    return result



def zmq_test():
    import zmq
    ctx = zmq.Context()
    # The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
    pupil_remote = ctx.socket(zmq.REQ)

    # ip = '127.0.0.1'  # If you talk to a different machine use its IP.
    ip = '192.168.0.31'  # If you talk to a different machine use its IP.
    port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

    pupil_remote.connect(f'tcp://{ip}:{port}')

    # Request 'SUB_PORT' for reading data
    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()
    print(sub_port)

    # Request 'PUB_PORT' for writing data
    pupil_remote.send_string('PUB_PORT')
    pub_port = pupil_remote.recv_string()
    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(f'tcp://{ip}:{sub_port}')
    subscriber.subscribe('pupil.1.3d')  # receive all gaze messages

    # we need a serializer
    import msgpack

    while True:
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        # print(f"{topic}: {message}")

def tcp_test():
    host = "192.168.0.22"
    port = 12345
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind((host, port))
    serverSocket.listen(1)
    print("WAITING FOR CONNECTION")
    connectionSocket, addr = serverSocket.accept()
    print(str(addr), 'connected')
    while True:
        data = connectionSocket.recv(1024)
        if data != None:
            print('received', data)
            connectionSocket.send(str(time.time()).encode("utf-8"))
            print('message sent', time.time())
    serverSocket.close()


def main():
    Pupil_thread = ZMQ_listener(name="Pupil Listener", args=[True])
    Pupil_thread.start()
    while True:
        pass

if __name__ == "__main__":
    main()
