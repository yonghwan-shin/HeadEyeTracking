import threading
import time
from time import sleep
import zmq

import os
import csv


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
		self.ZMQ_DATA_ROOT = ""
		self.ZMQ_SUBJECT = 0
		self.ZMQ_CURRFILE = ""
		self.ZMQ_RECORDING = ""
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
				if self.ZMQ_RECORDING == "RECORD":
					savefile_ZMQ(self, self.string2send)

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

def savefile_ZMQ(self, data):
	if os.path.isfile(
			self.ZMQ_DATA_ROOT + "/" + self.ZMQ_SUBJECT + "/" + "EYE_" + self.ZMQ_CURRFILE + ".csv"):  # Path location 할당해줘야합니다. 초기 헤더값 이후 데이터 어펜드.
		f = open(self.ZMQ_DATA_ROOT + "/" + self.ZMQ_SUBJECT + "/" + "EYE_" + self.ZMQ_CURRFILE + ".csv", 'a')
		wr = csv.writer(f, lineterminator='\n')
		# stop = timeit.default_timer()
		ts = time.time()
		current_time = []
		current_time.append(ts)
		wr.writerow(data + current_time)
	else:  # 초기 헤더값 설정
		f = open(self.ZMQ_DATA_ROOT + "/" + self.ZMQ_SUBJECT + "/" + "EYE_" + self.ZMQ_CURRFILE + ".csv", 'w')
		wr = csv.writer(f, lineterminator='\n')
		# trial_start_time = timeit.default_timer()
		wr.writerow(["zmq_X", "zmq_Y", "phi", "theta", "zmq_confidence", "ZMQtimestamp"])



if __name__ == "__main__":
	ZMQ_listener_main()

# print(message)
