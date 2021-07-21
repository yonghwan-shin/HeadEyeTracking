# import socketserver
#
#
# class TCPHandler(socketserver.BaseRequestHandler):
#     def handle(self):
#         self.data = self.request.recv(1024).strip()
#         print("{} wrote:".format(self.client_address[0]))
#         print(self.data)
#         self.request.sendall(self.data.upper())
#
#
# if __name__ == "__main__":
#     HOST, PORT = "localhost", 9051
#
#     with socketserver.TCPServer((HOST, PORT), TCPHandler) as server:
#         server.serve_forever()
import threading
import socket
import select


class TCP_server(threading.Thread):
    queue = None

    def __del__(self):
        if threading.Thread.is_alive(self):
            self.join()
        print("TCP server dead")

    def __init__(self, args=[], name='TCP Server'):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.name = name
        self.args = args
        self.port = 9051
        self.recording = False
        global notifier
        notifier = args[0]
        print(self.name, 'created')

    def set_queue(self, queue):
        self.queue = queue

    def bind_device(self):
        self.sock.bind(("", self.port))
        self.sock.listen(10)
        self.connection, self.address = self.sock.accept()
        print('device connected', self.address)

    def send_tcp_message(self, msg):
        try:
            self.connection.sendall(msg.encode('utf-8'))
        except Exception as e:
            print('error while sending TCP message', e.args)

    def trial_start_flag(self, event):
        event.set()

    def trial_end_flag(self, event):
        event.set()

    def run(self):
        print(threading.current_thread().getName(), "is started")
        self.bind_device()
        while True:
            try:
                # ready_to_read, ready_to_write, in_error = select.select([self.connection], [], [])
                # if self.connection in ready_to_read:
                buffer = self.connection.recv(1024)
                if not buffer: continue
                print('received', buffer)
                self.interpret_message(buffer)


            except Exception as e:
                self.connection.close()
                print('closing socket', e.args)
                break
        self.connection.close()

    def interpret_message(self, msg):
        if type(msg) == type(b'\n'):
            msg = msg.decode('utf-8')
        if 'TRIAL_START' in msg:
            self.recording = True
            print('interpreting.. START')
            notifier.raise_event("TRIAL_START", "TRIAL_START")
        elif 'TRIAL_END' in msg:
            self.recording = False
            notifier.raise_event("TRIAL_END", "TRIAL_END")
