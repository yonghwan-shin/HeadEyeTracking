import os.path
import serial.tools.list_ports
from zmq_pupil_origin import *
from Naming import *


def main():
    sub_num = int(input('Enter the subject number (1~999): '))
    # checkDirectory(DATA_ROOT)

    Pupil_thread = ZMQ_listener(name="Pupil Listener", args=[True])
    Pupil_thread.Set_sub_num(sub_num)
    Pupil_thread.start()

    while True:
        if Pupil_thread.Holo == None:
            print("re-finding hololens port")
            Pupil_thread.Holo = find_holo_serial_port("/dev/cu.Bluetooth")
            Pupil_thread.Holo.flush()
            continue
        Pupil_thread.buffer += Pupil_thread.Holo.read(Pupil_thread.Holo.in_waiting).decode()
        while "\n" in Pupil_thread.buffer:
            data, Pupil_thread.buffer = Pupil_thread.buffer.split("\n", 1)
            if data.startswith("#"):
                Pupil_thread.holodata.append(data)
            if data=="CALLBACK":
                print('comm delay: ',time.time() - Pupil_thread.delay_timer)

            print('received :', data)
        Pupil_thread.Holo_START()
        Pupil_thread.Holo_INIT()
        # TRIAL -> file save
        Pupil_thread.Holo_TRIAL()
        # END -> NEXT or BREAK or FINISH, NEXT: INIT 상태로 만들기.
        Pupil_thread.Holo_END()
        pass


if __name__ == "__main__":
    main()
