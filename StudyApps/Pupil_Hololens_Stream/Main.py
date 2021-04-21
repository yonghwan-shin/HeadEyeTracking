import os.path
import serial.tools.list_ports
from zmq_pupil_origin import *
from Naming import *
import sys
import socket
import time
from UDPcommunication import UDP_listener, UDP_sender, Queue

from multiprocessing import Pool


def run_tasks(function, args, pool):
    results = pool.map(function, args)
    return results


def Initialize_threads():
    sub_num = int(input('Enter the subject number (1~999): '))
    eye_recv_q = Queue()
    Pupil_thread = ZMQ_listener(q=eye_recv_q, name="Pupil Listener", args=[True])
    Pupil_thread.Set_sub_num(sub_num)
    Pupil_thread.start()
    udp_recv_q = Queue()
    UDP_Listener_thread = UDP_listener(q=udp_recv_q, name='UDP listener', args=[True])
    UDP_Listener_thread.Set_port(3000)
    UDP_Listener_thread.start()

    udp_send_q = Queue()
    UDP_Sender_thread = UDP_sender(q=udp_send_q, name='UDP sender', args=[True])
    UDP_Sender_thread.Set_destination("192.168.0.9", 5005)
    UDP_Sender_thread.start()
    return Pupil_thread, UDP_Listener_thread, UDP_Sender_thread, eye_recv_q, udp_recv_q, udp_send_q


def eye_handler(recv_q, send_q):
    while True:
        data, evt = recv_q.get()
        # do something
        send_q.put((data, evt))
        # evt.set()
        recv_q.task_done()


def UDP_test():
    Pupil_thread, UDP_Listener_thread, UDP_Sender_thread, eye_recv_q, udp_recv_q, udp_send_q = Initialize_threads()
    eye_thread = threading.Thread(target=eye_handler, args=(eye_recv_q, udp_send_q))
    eye_thread.start()
    while True:
        pass
    q.join()


def main():
    sub_num = int(input('Enter the subject number (1~999): '))
    # checkDirectory(DATA_ROOT)

    Pupil_thread = ZMQ_listener(name="Pupil Listener", args=[True])
    Pupil_thread.Set_sub_num(sub_num)
    Pupil_thread.start()

    a = 1
    # send_sock.close()
    callback = True
    callback_times = []
    lost_count = 0
    while True:
        # if Pupil_thread.Holo == None:
        #     print("re-finding hololens port")
        #     Pupil_thread.Holo = find_holo_serial_port("/dev/cu.Bluetooth")
        #     Pupil_thread.Holo.flush()
        #     continue
        # Pupil_thread.buffer += Pupil_thread.Holo.read(Pupil_thread.Holo.in_waiting).decode()
        # while "\n" in Pupil_thread.buffer:
        #     data, Pupil_thread.buffer = Pupil_thread.buffer.split("\n", 1)
        #     if data.startswith("#"):
        #         Pupil_thread.holodata.append(data)
        #     if data=="CALLBACK":
        #         print('comm delay: ',time.time() - Pupil_thread.delay_timer)
        #
        #     print('received :', data)
        # Pupil_thread.Holo_START()
        # Pupil_thread.Holo_INIT()
        # # TRIAL -> file save
        # Pupil_thread.Holo_TRIAL()
        # # END -> NEXT or BREAK or FINISH, NEXT: INIT 상태로 만들기.
        # Pupil_thread.Holo_END()
        # try:
        #     if callback==True:
        #         a +=1
        #         msg = bytes(str(a),'utf-8')
        #         send_time = time.time()
        #         send_sock.sendto(msg, destination)
        #         callback=False
        #         # print('sending',a)
        # except Exception as e:
        #     msg = bytes(str(a), 'utf-8')
        #     send_time = time.time()
        #     send_sock.sendto(msg, destination)
        #     callback = False
        #     # print('re-sending', a)
        #     print(e)
        #
        # try:
        #     data, sender = receive_sock.recvfrom(data_size)
        # except Exception as e:
        #     lost_count+=1
        #     msg = bytes(str(a), 'utf-8')
        #     send_time = time.time()
        #     send_sock.sendto(msg, destination)
        #     callback = False
        #     # print('re-sending', a)
        #     print(e)
        #
        # if a==int(data):
        #     # print(a,time.time()-send_time)
        #     callback_times.append(time.time()-send_time)
        #     callback=True
        #     print(a,1000*sum(callback_times)/len(callback_times), 1000*max(callback_times),1000*min(callback_times),1000*callback_times[-1],lost_count)

        pass


if __name__ == "__main__":
    UDP_test()
