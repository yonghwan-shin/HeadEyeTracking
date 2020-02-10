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

sub = 0
dataline = [0.0, 0.0, 0.0, 0.0, 0.0]
holodata = ['#START']
filename = []
timer = 0
# curr_file = []

def threadcontrol(): #init
    zmq_thread.start()
    imu_thread.start()

def checkDirectory( path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def connectHolo(): #init
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.device.startswith('/dev/cu.Bluetooth'):
            return serial.Serial(port.device, 115200)

def Holo_data_receive():
    signal = Holo.read(Holo.in_waiting)
    if signal.decode("utf-8") != False:
        if signal.decode("utf-8").startswith('#'):
            holodata.append(signal.decode("utf-8"))
        print('received :', signal.decode("utf-8"))

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
        print('remain trials:', len(filename))
        Holo_encoder("#NEXT_" + curr_file)
        holodata.append("#INIT")

def Holo_TRIAL():
    if (holodata[-1] == "#TRIAL"):
        imu_thread.Start_trial()
        zmq_thread.Start_trial()
        global timer
        timer = timeit.default_timer()

def Holo_END():
    if (holodata[-1] == "#END"):
        imu_thread.End_trial()
        zmq_thread.End_trial()
        print('trial takes', timeit.default_timer() - timer, 'seconds')
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
    filename = make_experiment_array(int(sub))
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
    while True:
        # print('Arduino：{}\n'.format(Serial_communication.dataline) + '\n' + 'Pupil : {}\n'.format(zmq_thread.string2send))
        if Holo.in_waiting > 0:
            Holo_data_receive()
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
