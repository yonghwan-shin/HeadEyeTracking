import time
import serial
import serial.tools.list_ports
print('serial ', serial.__version__)

# set a port number & baud rate
list = serial.tools.list_ports.comports()
PORT = ""
BaudRate = 115200
# print(list)
for element in list:
	print("PORT:", str(element.device))

for port, desc, hwid in sorted(list):
	print(f"{port}: {desc} {hwid}")
	if "Bl" in port:
		PORT = port
#
Serial = serial.Serial(PORT,BaudRate)
Serial.write("hello".encode("UTF-8"))
#
# while(True):
# 	info = "#0.1$0.9$0.9$@"
# 	# Serial.write(info)
# 	Serial.write(info.encode('UTF-8'))
# 	time.sleep(0.05)
# 	# info = "#0.9$0.1$0.9$"
# 	# # Serial.write(info)
# 	# Serial.write(info.encode('UTF-8'))
# 	# time.sleep(0.05)
# 	print(Serial.read(Serial.inWaiting()))

# if __name__ == "__main__":

