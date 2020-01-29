import sys
import csv
import os.path
import timeit
import time

import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from zmq_pupil import *
from Naming import *

zmq_thread = ZMQ_listener(name='ZMQ_listener', args=[True])


class MyAppGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.cb_port = QComboBox()
        self.cb_port2 = QComboBox()
        self.sub_number = QComboBox()

        self.qtxt1 = QTextEdit(self)
        self.qtxt2 = QTextEdit(self)
        self.btn4 = QPushButton('Port_Connect', self)
        self.btn1 = QPushButton("Start", self)
        self.btn2 = QPushButton("Stop", self)
        self.btn3 = QPushButton('Quit', self)

        vbox = QVBoxLayout()

        vbox.addWidget(QLabel(self.tr("Arduino Port")))
        vbox.addWidget(self.cb_port)
        vbox.addWidget(QLabel(self.tr("Hololens Port")))
        vbox.addWidget(self.cb_port2)
        vbox.addWidget(QLabel(self.tr("Subject Number")))
        vbox.addWidget(self.sub_number)

        vbox.addWidget(self.qtxt1)
        vbox.addWidget(self.qtxt2)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btn3)

        self.setLayout(vbox)
        self._fill_serial_info()

        self.setWindowTitle('Head_Eye_Tracking')


class MyApp(MyAppGUI):
    port_signal = pyqtSignal(str)
    port2_signal = pyqtSignal(str)
    sub = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.btn1.clicked.connect(self.exp_start)
        self.btn2.clicked.connect(self.exp_stop)
        self.btn3.clicked.connect(self.exp_quit)
        self.btn4.clicked.connect(self.send_port_signal)

        self.th = MyAppThread(parent=self)

        self.th.viewer.connect(self.exp_update)
        self.th.recording.connect(self.rec_update)

        self.port_signal.connect(self.th.receive_port_singal)
        self.port2_signal.connect(self.th.receive_port2_singal)
        self.sub.connect(self.th.sub_singal)

        self.show()

    def _fill_serial_info(self):
        self.cb_port.insertItems(0, [str(x[0]) for x in self._get_available_port()])
        self.cb_port2.insertItems(0, [str(x[0]) for x in self._get_available_port()])
        self.sub_number.insertItems(0, [str(x) for x in range(1, 50)])

    def _get_available_port(self):
        available_port = serial.tools.list_ports.comports()
        return available_port

    @pyqtSlot()
    def exp_start(self):
        self.qtxt2.append('Experiment Start')
        self.th.start()
        self.th.working = True

    @pyqtSlot()
    def exp_stop(self):
        self.qtxt2.append('Experiment Stop')
        self.th.working = False

    @pyqtSlot()
    def exp_quit(self):
        ex.destroy()
        sys.exit()

    @pyqtSlot()
    def send_port_signal(self):
        subnumber = self.sub_number.currentText()
        self.sub.emit(subnumber)
        self.qtxt2.append('Arduino connected')
        self.qtxt2.append('HOLO connected')
        zmq_thread.start()
        self.qtxt2.append('Zmq started')
        global arduino
        arduino = serial.Serial(self.cb_port.currentText(), 9600)
        global Holo
        Holo = serial.Serial(self.cb_port2.currentText(), 115200)

    @pyqtSlot(str)
    def exp_update(self, msg):
        self.qtxt1.clear()
        self.qtxt1.append(msg)

    def rec_update(self, msg):
        if msg == "Recording":
            self.qtxt2.setStyleSheet("color: rgb(200, 0, 0);")


class MyAppThread(QThread):
    viewer = pyqtSignal(str)
    recording = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.working = False
        global holodata
        holodata = ["#START"]

    def __del__(self):
        print(".... end thread.....")
        self.wait()

    def view(self):
        if (arduino.in_waiting > 0):
            res = arduino.readline()
            global dataline
            dataline = res.decode()[:len(res) - 3].split(',')
            self.viewer.emit('Arduino：{}'.format(dataline) + '\n' + 'Pupil : {}'.format(zmq_thread.string2send))

    def Holo_encoder(self,string):
        Holo.write(string.encode("UTF-8"))

    def Holo_data_receive(self):
        signal = Holo.read(Holo.in_waiting)
        if signal.decode("utf-8") != False:
            holodata.append(signal.decode("utf-8"))

    def Holo_START(self):
        if (holodata[-1] == "#START"):
            self.Holo_encoder("#SUB" + str(self.sub))
        sleep(1)


    def Holo_INIT(self):
        if ((holodata[-1] == "#INIT") and (holodata[-2] != "#INIT")):
            curr_file = current_add(filename.pop())
            self.Holo_encoder("#NEXT_" + curr_file)
            holodata.append("#INIT")

    def Holo_TRIAL(self):
        if (holodata[-1] == "#TRIAL"):
            self.recording.emit("Recording")
            if os.path.isfile(
                    "/Users/Jiwan/Documents/GitHub/HeadEyeTracking/StudyApps/Pupil_study_app/%.csv" % curr_file):  # Path location 할당해줘야합니다. 초기 헤더값 이후 데이터 어펜드.
                f = open('%.csv' % curr_file, 'a')
                wr = csv.writer(f, lineterminator='\n')
                stop = timeit.default_timer()
                wr.writerow(dataline + zmq_thread.string2send + [stop - start])
            else:  # 초기 헤더값 설정
                f = open('%.csv' % curr_file, 'w')
                wr = csv.writer(f, lineterminator='\n')
                start = timeit.default_timer()
                wr.writerow(["quatI", "quatJ", "quatK", "quatReal", "quatRadianAccuracy", "zmq_X", "zmq_Y",
                             "zmq_confidence", "IMUtimestamp"])

    def Holo_END(self):
        if (holodata[-1] == "#END"):
            if (filename[-1] == 'BREAK'):
                self.Holo_encoder("#BREAK")
            elif (filename[-1] == 'FINISH'):
                self.Holo_encoder("#FINISH")
            else:
                holodata.append("INIT")

    def run(self):
        while self.working:
            # print current data in viewer
            self.view()
            if (Holo.in_waiting > 0):
                # Holo signal read
                self.Holo_data_receive()
                # connect -> subject send # send sub number 안받으면 쌓여서 에러남.
                self.Holo_START()
                # INIT -> curr_file send
                self.Holo_INIT()
                # TRIAL -> file save
                self.Holo_TRIAL()
                # END -> NEXT or BREAK or FINISH, NEXT: INIT 상태로 만들기.
                self.Holo_END()

    @pyqtSlot(str)
    def receive_port_singal(self, inst):
        self.port = inst

    @pyqtSlot(str)
    def receive_port2_singal(self, inst):
        self.port2 = inst

    @pyqtSlot(str)
    def sub_singal(self, inst):
        self.sub = inst
        global filename
        filename = make_experiment_array(int(self.sub))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
