# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from TcpCommunication import TCP_server

def main_experiment():
    sub_num = int( input('type subject number:'))
    print('subject number set to : ',sub_num)
    TCP =TCP_server()
    TCP.start()
    TCP.send_tcp_message("subject_"+str(sub_num))
    while True:
        pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_experiment()

