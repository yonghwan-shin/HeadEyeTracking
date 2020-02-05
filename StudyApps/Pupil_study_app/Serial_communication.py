import time
import serial
import serial.tools.list_ports
import threading

import os
import csv

connected = False

arduino = serial.Serial()

dataline = [0.0,0.0,0.0,0.0,0.0]

IMU_DATA_ROOT =""
IMU_SUBJECT=0
IMU_CURRFILE=""
IMU_RECORDING=""

def connectArduino():
	ports = serial.tools.list_ports.comports()
	print('reconnecting')
	for port in ports:
		# print(port.device)
		if port.device.startswith('/dev/cu.usbmodem'):
			return serial.Serial(port.device, 9600);


# def handle_data(data):
# 	print(data)
# 	pass

def read_from_arduino(port):
	while True:
		try:
			if port ==None:
				port =connectArduino()

			res = port.readline()
			global dataline
			dataline = res.decode()[:len(res) - 3].split(',')
			if IMU_RECORDING == "RECORD":
				savefile_arduino(dataline)


		except:
			if not port == None:
				port.close()
				port = None
				print('disconnecting')
				dataline = [0.0,0.0,0.0,0.0,0.0]
			time.sleep(2)

def savefile_arduino(data):
	if os.path.isfile(
			IMU_DATA_ROOT + "/" + IMU_SUBJECT + "/" + "IMU_"+ IMU_CURRFILE + ".csv"):  # Path location 할당해줘야합니다. 초기 헤더값 이후 데이터 어펜드.
		f = open(IMU_DATA_ROOT + "/" + IMU_SUBJECT + "/" + "IMU_"+ IMU_CURRFILE + ".csv", 'a')
		wr = csv.writer(f, lineterminator='\n')
		# stop = timeit.default_timer()
		ts = time.time()
		current_time = []
		current_time.append(ts)
		wr.writerow(data + current_time)
	else:  # 초기 헤더값 설정
		f = open(IMU_DATA_ROOT + "/" + IMU_SUBJECT + "/" + "IMU_"+ IMU_CURRFILE + ".csv", 'w')
		wr = csv.writer(f, lineterminator='\n')
		# trial_start_time = timeit.default_timer()
		wr.writerow(["quatI", "quatJ", "quatK", "quatReal", "quatRadianAccuracy", "IMUtimestamp"])



if __name__ == "__main__":
	arduino = connectArduino()
	arduinoThread = threading.Thread(target = read_from_arduino, args = (arduino,))
	arduinoThread.start()