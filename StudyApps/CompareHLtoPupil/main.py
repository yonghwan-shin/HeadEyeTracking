import threading
from PupilLabs import ZMQ_listener

from TcpCommunication import TCP_server

stop_event = threading.Event()
start_event = threading.Event()

from queue import Queue
from EventNotifier import Notifier


def main_experiment():
    notifier = Notifier(["TRIAL_START", "TRIAL_END"])
    TCP = TCP_server(args=(notifier,))
    TCP.start()
    pupil = ZMQ_listener(args=(notifier,))
    pupil.start()

    notifier.subscribe("TRIAL_START", pupil)
    notifier.subscribe("TRIAL_END", pupil)
    while True:
        pass


if __name__ == '__main__':
    main_experiment()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
