import sys
import os
import time

from FileHandling import *
from SerialCommunication import *
from FileHandling import *
from zmq_pupil_labs import *

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
subject = input("Participant number? :")
RUNNING = True


# global timer, now


# arduino_listener = IMU_listener()

def send_to_hololens(message):
    hololens.write(message.encode("UTF-8"))
    print("Send to Hololens : ", message.encode("UTF-8"))


def initialize_connections():
    global imu, pupil, hololens, timer, now, holo_buffer, holo_signals, current_trial, trials
    imu = ImuListener()
    imu.start();
    sleep(0.3)
    pupil = PupilListener()
    pupil.start();
    sleep(0.3)

    hololens = connect_hololens()
    timer = time.time()
    now = time.time()
    holo_buffer = []
    holo_signals = ['#START']
    current_trial = 0
    trials = make_trial_sequence(1)
    # return imu_thread, pupil_thread


def handle_hololens_signal(t):
    if holo_signals[-1] == '#START':
        send_to_hololens("#SUB" + str(subject))
        sleep(1)
    elif (holo_signals[-1] == "#INIT") and (holo_signals[-2] != '#INIT'):
        imu.set_filename(trials[t])
        pupil.set_filename(trials[t])
        print(f'current trial: {t}')
        send_to_hololens("#NEXT_" + trials[t])
        t = t + 1
    elif (holo_signals[-1] == '#TRIAL') and (holo_signals[-2] != "#TRIAL"):
        imu.start_trial()
        pupil.start_trial()
        global timer
        timer = time.time()
        holo_signals.append()
    elif holo_signals[-1] == '#END':
        # global timer
        imu.end_trial()
        pupil.end_trial()
        # global now
        print(f'trial takes {now - timer} seconds')
        if trials[t] == 'BREAK' and holo_signals[-2] != 'BREAK':
            send_to_hololens("BREAK")
            t = t + 1
            holo_signals.append("BREAK")
        elif trials[t] == 'FINISH' and holo_signals[-2] != 'FINISH':
            send_to_hololens("#FINISH")
            t = t + 1
            holo_signals.append("FINISH")
        else:
            holo_signals.append("#INIT")
    return t


def loop():
    initialize_connections()

    check_directory(DATA_ROOT)
    check_directory(DATA_ROOT + "/" + str(subject))
    global now
    while RUNNING:

        now = time.time()
        if hololens.in_waiting > 0:
            holo_buffer.append(hololens.read(hololens.in_waiting).decode('utf-8'))
            for component in holo_buffer:
                if component.startswith('#'):
                    holo_signals.append(component)
                print(f'received: {component}')
            holo_buffer = []
        current_trial = handle_hololens_signal(current_trial)


# from itertools import cycle
# from time import sleep
#
# for frame in cycle(r'-\|/-\|/'):
#     print('\r', frame,'hello' ,sep='', end=' ', flush=True)
#
#     sleep(0.5)

if __name__ == '__main__':
    loop()

    sys.exit()  # Experiment Finished
