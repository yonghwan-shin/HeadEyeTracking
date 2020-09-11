from zmq_pupil import *

def main():
    Pupil_thread = ZMQ_listener(name='Pupil Listener',args = [True])
    Pupil_thread.start()
    while(True):
        pass


if __name__ == '__main__':
    main()
