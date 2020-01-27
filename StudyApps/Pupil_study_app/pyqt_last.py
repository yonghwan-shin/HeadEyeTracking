import sys

import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from zmq_pupil import *

zmq_thread = ZMQ_listener(name='ZMQ_listener', args=[True])


class MyAppGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.cb_port = QComboBox()
        self.sub_number = QComboBox()

        self.qtxt1 = QTextEdit(self)
        self.qtxt2 = QTextEdit(self)
        self.btn4 = QPushButton('Port_Connect', self)
        self.btn1 = QPushButton("Start", self)
        self.btn2 = QPushButton("Stop", self)
        self.btn3 = QPushButton('Quit', self)

        vbox = QVBoxLayout()

        vbox.addWidget(QLabel(self.tr("Port")))
        vbox.addWidget(self.cb_port)
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

    def __init__(self,parent=None):
        super().__init__(parent)

        self.btn1.clicked.connect(self.exp_start)
        self.btn2.clicked.connect(self.exp_stop)
        self.btn3.clicked.connect(self.exp_quit)
        self.btn4.clicked.connect(self.send_port_signal)

        self.th = MyAppThread(parent=self)

        self.th.sec_changed.connect(self.exp_update)

        self.port_signal.connect(self.th.receive_port_singal)

        self.show()

    def _fill_serial_info(self):
        self.cb_port.insertItems(0, [str(x[0]) for x in self._get_available_port()])
        self.sub_number.insertItems(0, [str(x) for x in range(1,50)])

    def _get_available_port(self):
        available_port = serial.tools.list_ports.comports()
        return available_port

    @pyqtSlot()
    def exp_start(self):
        self.qtxt2.append('Experiment Start')
        self.serial = serial.Serial(self.cb_port.currentText(), 9600)
        self.th.start()
        self.th.working = True

    @pyqtSlot()
    def exp_stop(self):
        self.qtxt2.append('Experiment Stop')
        self.serial.close()
        self.th.working = False

    @pyqtSlot()
    def exp_quit(self):
        ex.destroy()
        sys.exit()

    @pyqtSlot()
    def send_port_signal(self):
        portname = self.cb_port.currentText()
        self.port_signal.emit(portname)
        self.qtxt2.append('Arduino connected')
        zmq_thread.start()
        self.qtxt2.append('Zmq connected')

    @pyqtSlot(str)
    def exp_update(self, msg):
        self.qtxt1.append(msg)


class MyAppThread(QThread):
    sec_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.working = True

    def __del__(self):
        print(".... end thread.....")
        self.wait()

    def run(self):
        while self.working:
            arduino = serial.Serial(self.port, 9600)
            if arduino.readable():
                res = arduino.readline()
                dataline = res.decode()[:len(res) - 3].split(',')
            self.sec_changed.emit('Arduino：{}'.format(dataline)+ '\n' + zmq_thread.string2send)

            self.sleep(1)

    @pyqtSlot(str)
    def receive_port_singal(self, inst):
        self.port = inst

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())