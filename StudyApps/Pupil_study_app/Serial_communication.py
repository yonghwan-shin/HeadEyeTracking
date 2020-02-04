import time
import serial
import serial.tools.list_ports
import threading

connected = False

arduino = serial.Serial()

dataline = [0.0,0.0,0.0,0.0,0.0]

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
			# print(dataline)

		except:
			if not port == None:
				port.close()
				port = None
				print('disconnecting')
				dataline = [0.0,0.0,0.0,0.0,0.0]
			time.sleep(2)



if __name__ == "__main__":
	arduino = connectArduino()
	arduinoThread = threading.Thread(target = read_from_arduino, args = (arduino,))
	arduinoThread.start()
