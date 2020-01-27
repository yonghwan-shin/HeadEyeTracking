import serial
import csv
import os.path
import timeit


port = '/dev/cu.usbmodem14201'
brate = 9600 #boudrate
SubNumber=15

check = 'start'
arduino = serial.Serial(port, 9600)

while True:
    if arduino.readable():
        res = arduino.readline()
        dataline = res.decode()[:len(res) - 3].split(',')

    if check == 'start':
        if os.path.isfile("/Users/Jiwan/Documents/GitHub/HeadEyeTracking/StudyApps/getQuats120/sub%d.csv"%SubNumber):
            f = open('sub%d.csv'%SubNumber, 'a')
            wr = csv.writer(f, lineterminator='\n')
            stop = timeit.default_timer()
            print(type(res))
            print(dataline)
            # wr.writerow(dataline + [stop-start])
        else:
            f = open('sub%d.csv'%SubNumber, 'w')
            wr = csv.writer(f, lineterminator='\n')
            start = timeit.default_timer()
            wr.writerow(["quatI","quatJ","quatK","quatReal","quatRadianAccuracy","IMUtimestamp"])
            stop = timeit.default_timer()
            print(type(res))
            print(dataline)
            # wr.writerow(dataline + [stop-start])

    #if 외부 통신장치 input이 있으면
    #   f.close()
    #   break