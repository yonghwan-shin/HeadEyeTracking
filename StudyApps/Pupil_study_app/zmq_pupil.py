import threading
from time import sleep
import zmq

ctx = zmq.Context()
pupil_remote = ctx.socket(zmq.REQ)
ip = 'localhost'
# ip = '192.168.0.49'
port = 50020
ZMQ_stop = False

def ZMQ_connect():
	try:
		pupil_remote.connect(f'tcp://{ip}:{port}')
		# Request 'SUB_PORT' for reading data
		pupil_remote.send_string('SUB_PORT')
		sub_port = pupil_remote.recv_string()

		# Request 'PUB_PORT' for writing data
		pupil_remote.send_string('PUB_PORT')
		pub_port = pupil_remote.recv_string()

		subscriber = ctx.socket(zmq.SUB)
		subscriber.connect(f'tcp://{ip}:{sub_port}')
		subscriber.subscribe('pupil.1')
	except KeyboardInterrupt:
		pass
	print('ZMQ start receiving')
	return subscriber


# pupil_remote.connect(f'tcp://{ip}:{port}')
# # Request 'SUB_PORT' for reading data
# pupil_remote.send_string('SUB_PORT')
# sub_port = pupil_remote.recv_string()
#
# # Request 'PUB_PORT' for writing data
# pupil_remote.send_string('PUB_PORT')
# pub_port = pupil_remote.recv_string()
#
# subscriber = ctx.socket(zmq.SUB)
# subscriber.connect(f'tcp://{ip}:{sub_port}')
# subscriber.subscribe('pupil.1')
# print('start receiving')


class ZMQ_listener(threading.Thread):

	def __del__(self):
		if threading.Thread.isAlive(self):
			self.join()
		print("ZMQ thread dead")

	def __init__(self, args, name="Pupil Listener"):
		threading.Thread.__init__(self)
		threading.Thread.daemon = True
		self.string2send = ['NaN','NaN','NaN','NaN','NaN']
		self.name = name
		self.args = args


	def run(self):
		# actual part
		print(threading.currentThread().getName(), " is started")
		subscriber = ZMQ_connect()
		sleep(1)
		import msgpack

		while self.args[0]:
			try:
				topic, payload = subscriber.recv_multipart()
				message = msgpack.unpackb(payload, encoding='utf-8')
				# string2send = "#"
				f1 = str(message['norm_pos'][0])
				f2 = str(message['norm_pos'][1])
				f3 = str(message['phi'])
				f4 = str(message['theta'])
				f5 = str(message['confidence'])
				global string2send
				self.string2send = [f1,f2,f3,f4,f5]

				# print(self.string2send)
				# send_message(bytes(string2send,'utf-8'))

				# print(message['norm_pos'],f3)
				# print("------------------------")
			except KeyboardInterrupt:
				break
		sleep(0.1)
		self.join()
		print('end')
		return


def ZMQ_listener_main():
	zmq_receiver_name = "ZMQ_listener"
	zmq_thread = ZMQ_listener(name='ZMQ_listener', args=(zmq_receiver_name,True))
	zmq_thread.start()



if __name__ == "__main__":
	ZMQ_listener_main()

# print(message)
