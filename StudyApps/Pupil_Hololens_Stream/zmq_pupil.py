import threading
import time
from time import sleep
import zmq
import serial
import serial.tools.list_ports
import os
import csv

ctx = zmq.Context()
pupil_remote = ctx.socket(zmq.REQ)
ip = 'localhost'
# ip = '192.168.0.49'
port = 50020
ZMQ_stop = False
def find_holo_serial_port(portname):
	ports = serial.tools.list_ports.comports()
	for port in ports:
		if port.device.startswith(portname):
			print(port.device)
			return serial.Serial(port.device,115200)

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
		if threading.Thread.is_alive(self):
			self.join()
		print("ZMQ thread dead")

	def __init__(self, args, name="Pupil Listener"):
		threading.Thread.__init__(self)
		threading.Thread.daemon = True
		self.DATA_ROOT =""
		self.sub_num = 0
		self.filename = ""
		self.recording = False
		self.string2send = ['NaN','NaN','NaN','NaN','NaN']
		self.name = name
		self.args = args
		self.stored_data = []
		
		self.buffer = ''
		self.timestapmes = []
		self.count  = 1000 
		self.sent = 1000 
		self.Holo = find_holo_serial_port('/dev/cu.Bluetooth')
		self.Holo.flush()
		self.send_time = 0
		self.delays = []

	def run(self):
		# actual part
		print(threading.currentThread().getName(), " is started")
		subscriber = ZMQ_connect()
		
		sleep(1)
		# print(self.Holo.name)
		import msgpack

		while self.args[0]:
			try:
				topic, payload = subscriber.recv_multipart()
				message = msgpack.unpackb(payload, encoding='utf-8')
				# string2send = "#"
				# f1 = str(message['norm_pos'][0])
				# f2 = str(message['norm_pos'][1])
				# f3 = str(message['phi'])
				# f4 = str(message['theta'])
				# f5 = str(message['confidence'])
				# global string2send
				# self.string2send = [f1,f2,f3,f4,f5]
				# print(message['timestamp'],message['norm_pos'])
				if self.Holo == None:
					print('finding holo')
					self.Holo = find_holo_serial_port('/dev/cu.Bluetooth')
					continue
				
				self.buffer += self.Holo.read(self.Holo.in_waiting).decode()
				while '\n' in self.buffer:
					data, self.buffer = self.buffer.split('\n',1)
					print(data)
					self.delays.append((float(time.time()) - float(self.send_time))*1000)
					if data == '#' + str(self.sent) + '#'*14 :
						print(
							'chat delay', 
							format(
								(float(time.time()) - float(self.send_time))*1000
								,'.2f'),
								'ms',
								'mean delay : ',
								sum(self.delays) / len(self.delays)
						)
						# self.sent += 1000
				# self.send_to_hololens(str(message['norm_pos'])+'\n')
				if self.count % 100 ==1: 
					self.send_time = time.time()
					self.sent = self.count
					self.send_to_hololens('#' + str(self.sent) + '#'*14 + '\n')
				# [0.3894290751263883, 0.11579204756622086]
				
				confidence = 'O' if message['confidence'] > 0.6 else 'X'

				send_message = '$' + confidence + format(message['norm_pos'][0],'.6f') +','+ format(message['norm_pos'][1],'.6f') +'\n'
				self.send_to_hololens(send_message)
				# print(send_message)
				self.count += 1
				if self.count > 9999: self.count = 1000
				
				# if self.recording:
				# 	self.stored_data.append([str(time.time()),str(message)])

					# self.timestapmes.append(time.time())
					# savefile_ZMQ(self, self.string2send)

			except KeyboardInterrupt:
				break
		sleep(0.1)
		self.join()
		return


            	# return serial.Serial(port.device, 115200)
	def send_to_hololens(self,msg:str):
		self.Holo.write(msg.encode('UTF-8'))
		
	
	def save_data(self):
		full_name = 'EYE_' + self.filename + '.csv'
		file_path = os.path.join(self.DATA_ROOT,str(self.sub_num),full_name)

		# if os.path.isfile(file_path):
		# 	f = open(file_path,'a')
		# else:
		f= open(file_path,'w')

		wr = csv.writer(f, lineterminator='\n')
		wr.writerow(['eye_packets',len(self.stored_data)])
		wr.writerow(["norm_X", "norm_Y", "phi", "theta", "confidence", "timestamp"])
		for line in self.stored_data:
			wr.writerow(line)
		f.close()
		print('saved', full_name, ' total', len(self.stored_data), 'eye packets')
		self.stored_data.clear()
	def End_trial(self):
		self.recording = False
		self.save_data()
		# self.stored_data.clear()	#just to be sure
	def Start_trial(self):
		# self.stored_data.clear()	#just to be sure
		self.recording = True
	def Set_filename(self,_filename):
		self.filename = _filename
	def Set_sub_num(self,_sub_num):
		self.sub_num = _sub_num

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
		wr.writerow(["zmq_X", "zmq_Y", "phi", "theta", "zmq_confidence", "ZMQtimestamp"])




if __name__ == "__main__":
	ZMQ_listener_main()
