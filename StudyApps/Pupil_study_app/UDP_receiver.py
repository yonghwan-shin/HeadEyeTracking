import threading
from time import sleep
import socket


def connect_listener():
	try:
		port = 12001
		# ip = "192.168.0.9"
		ip = "192.168.0.9"
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		# sock.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
		sock.bind(("0.0.0.0", port))
	except:
		pass
	return sock


class UDP_listener(threading.Thread):
	def __init__(self, args, name="UDP listener"):
		threading.Thread.__init__(self)
		threading.Thread.daemon = True
		self.name = name
		self.args = args

	def run(self):
		# actual part whe run start()
		print(threading.currentThread().getName(), " is started")

		sleep(1)
		sock = connect_listener()
		while True:
			try:
				data, addr = sock.recvfrom(1024)
				print("received ", data, "add :", addr)
				pass
			except socket.error as msg:
				print(msg)
				pass
			except KeyboardInterrupt:
				break
		sleep(0.1)
		self.join()
		print("end")
		return


def UDP_listener_main():
	th = UDP_listener(name='UDP listener')
	th.start()


if __name__ == "__main__":
	UDP_listener_main()
