import os.path
import serial.tools.list_ports
from zmq_pupil import *
from Naming import *
from Serial_communication import *
import timeit
import Serial_communication

zmq_thread = ZMQ_listener(name='ZMQ_listener', args=[True])
imu_thread = IMU_listener(name='IMU_listener', args=[True])

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
zmq_thread.DATA_ROOT = DATA_ROOT
imu_thread.DATA_ROOT = DATA_ROOT

sub = 211

holodata = ['#START']
restart = False

# restart_number = 49
# restart = True
# holodata = ['#START','#INIT']


filename = []
# global timer
timer = 0
now = 0
global holo_buffer
holo_buffer = []


# curr_file = []

def threadcontrol():  # init
    zmq_thread.start()
    imu_thread.start()


def checkDirectory(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def connectHolo():  # init
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # if port.device.startswith('/dev/cu.Bluetooth'):
        if port.device.startswith('/dev/cu.DESKTOP'):
            return serial.Serial(port.device, 115200)


def Holo_data_receive(buffer):
    buffer += Holo.read(Holo.in_waiting).decode()
    while '\n' in buffer:
        data, buffer = buffer.split('\n', 1)
        if data.startswith('#'):
            holodata.append(data)
        print('received:', data)

    # signal = Holo.read(Holo.in_waiting)
    #
    # if signal.decode("utf-8") != False:
    #     if signal.decode("utf-8").startswith('#'):
    #         holodata.append(signal.decode("utfd-8"))
    #     print('received :', signal.decode("utf-8"))


def Holo_START():
    if (holodata[-1] == "#START"):
        Holo_encoder("#SUB" + str(sub))
        sleep(2)


def Holo_INIT():
    if ((holodata[-1] == "#INIT") and (holodata[-2] != "#INIT")):
        global curr_file
        curr_file = current_add(filename.pop())
        zmq_thread.Set_filename(curr_file)
        imu_thread.Set_filename(curr_file)
        print('remaining trials:', len(filename))
        Holo_encoder("#NEXT_" + curr_file)
        holodata.append("#INIT")


def Holo_TRIAL():
    if (holodata[-1] == "#TRIAL") and (holodata[-2] != "#TRIAL"):
        imu_thread.Start_trial()
        zmq_thread.Start_trial()
        global timer
        timer = time.time()
        holodata.append("#TRIAL")


def Holo_END():
    if (holodata[-1] == "#END"):
        global timer
        global now
        imu_thread.End_trial()
        zmq_thread.End_trial()
        now = time.time()
        print('-'*20,'END','-'*20,end=' ')
        
        print('time:', "%.3f" % (now - timer), 'sec')
        if (filename[-1] == 'BREAK' and (holodata[-2] != "BREAK")):
            Holo_encoder("#BREAK")
            filename.pop()
            holodata.append('BREAK')
        elif (filename[-1] == 'FINISH' and (holodata[-2] != "FINISH")):
            Holo_encoder("#FINISH")
            filename.pop()
            holodata.append('FINISH')
        else:
            holodata.append("#INIT")


def sub_singal():  # init
    imu_thread.Set_sub_num(str(sub))
    zmq_thread.Set_sub_num(str(sub))
    print('subject number was set as', str(sub))
    global filename
    filename = make_experiment_array_walkonly(int(sub))

    if restart:
        for _ in range(restart_number):
            filename.pop()
    checkDirectory(DATA_ROOT + "/" + str(sub))


def Holo_encoder(string):
    Holo.write(string.encode("UTF-8"))
    print("write to hololens : ", string.encode("UTF-8"))


def setup():
    #     connect hololens
    global sub
    sub = (input("enter subject number: "))
    print('subject number was set to : ' + sub)
    global Holo
    Holo = connectHolo()
    threadcontrol()
    checkDirectory(DATA_ROOT)
    sub_singal()


def loop():
    global holo_buffer
    # global Holo   
    while True:
        if Holo.in_waiting > 0:
            holo_buffer.append(Holo.read(Holo.in_waiting).decode('utf-8'))
            for component in holo_buffer:
                if component.startswith('#'):
                    holodata.append(component)
                print(f'received: {component}')
            holo_buffer = []
        # Holo_START(): INIT -> curr_file send
        Holo_START()
        Holo_INIT()
        # TRIAL -> file save
        Holo_TRIAL()
        # END -> NEXT or BREAK or FINISH, NEXT: INIT 상태로 만들기.
        Holo_END()



def main():
    setup()
    loop()


if __name__ == "__main__":
    main()
