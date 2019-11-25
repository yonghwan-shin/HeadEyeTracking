import socket
import time

sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sender.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
# Set a timeout so the socket does not block
# indefinitely when trying to receive data.
sender.settimeout(0.2)
sender.bind(("", 12001))
print("UDP sender socket started")
# message = b"your very important message"


def send_message(message):
	sender.sendto(message, ('192.168.0.9', 30000))
# while True:
# 	# server.sendto(message, ('<broadcast>', 30000))
# 	server.sendto(message, ('192.168.0.9', 30000))
# 	print("message sent!")
# 	time.sleep(1)
#
# while True:
# 	pass
