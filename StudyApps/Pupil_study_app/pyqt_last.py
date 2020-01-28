import sys

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

    def __init__(self,parent=None):
        super().__init__(parent)

        self.btn1.clicked.connect(self.exp_start)
        self.btn2.clicked.connect(self.exp_stop)
        self.btn3.clicked.connect(self.exp_quit)
        self.btn4.clicked.connect(self.send_port_signal)

        self.th = MyAppThread(parent=self)

        self.th.viewer.connect(self.exp_update)

        self.port_signal.connect(self.th.receive_port_singal)
        self.port2_signal.connect(self.th.receive_port2_singal)
        self.sub.connect(self.th.sub_singal)

        self.show()

    def _fill_serial_info(self):
        self.cb_port.insertItems(0, [str(x[0]) for x in self._get_available_port()])
        self.cb_port2.insertItems(0, [str(x[0]) for x in self._get_available_port()])
        self.sub_number.insertItems(0, [str(x) for x in range(1,50)])

    def _get_available_port(self):
        available_port = serial.tools.list_ports.comports()
        return available_port

    @pyqtSlot()
    def exp_start(self):
        self.qtxt2.append('Experiment Start')
        self.serial = serial.Serial(self.cb_port.currentText(), 9600)
        self.serial2 = serial.Serial(self.cb_port2.currentText(), 9600)
        self.th.start()
        self.th.working = True

    @pyqtSlot()
    def exp_stop(self):
        self.qtxt2.append('Experiment Stop')
        self.serial.close()
        self.serial2.close()
        self.th.working = False

    @pyqtSlot()
    def exp_quit(self):
        ex.destroy()
        sys.exit()

    @pyqtSlot()
    def send_port_signal(self):
        portname = self.cb_port.currentText()
        self.port_signal.emit(portname)
        port2name = self.cb_port2.currentText()
        self.port2_signal.emit(port2name)
        subnumber = self.sub_number.currentText()
        self.sub.emit(subnumber)
        self.qtxt2.append('Arduino connected')
        self.qtxt2.append('HOLO connected')
        zmq_thread.start()
        self.qtxt2.append('Zmq started')

    @pyqtSlot(str)
    def exp_update(self, msg):
        self.qtxt1.clear()
        self.qtxt1.append(msg)


class MyAppThread(QThread):
    viewer = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.working = True

    def __del__(self):
        print(".... end thread.....")
        self.wait()

    def run(self):
        while self.working:
            # arduino, zmq read
            arduino = serial.Serial(self.port, 9600)
            if arduino.readable():
                res = arduino.readline()
                dataline = res.decode()[:len(res) - 3].split(',')
            self.viewer.emit('Arduino：{}'.format(dataline)+ '\n' + 'Pupil : {}'.format(zmq_thread.string2send))

            # Holo connect / sub send
            Holo = serial.Serial(self.port2, 115200)
            Holo.write(("#SUB"+str(self.sub)).encode("UTF-8"))

            # HOLO read
            # if Holo.readable():
            #     signal = Holo.readline() #<< 여기서 에러 발생 print 해보고 알맞게 수정 필요.
            #     holodata = signal.decode()
            #     # holodata = "#TRIAL"
            #
            # # Experiment
            # if (holodata == "#TRIAL"):
            #     self.viewer.emit("Record Starting")
            #




            # self.sleep(1)

    @pyqtSlot(str)
    def receive_port_singal(self, inst):
        self.port = inst

    @pyqtSlot(str)
    def receive_port2_singal(self, inst):
        self.port2 = inst

    @pyqtSlot(str)
    def sub_singal(self, inst):
        self.sub = inst

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())