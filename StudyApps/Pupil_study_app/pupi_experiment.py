from zmq_pupil import *
from UDP_sender import *
from UDP_receiver import *

from tkinter import *
import sys


class Checkbar(Frame):
	def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
		Frame.__init__(self, parent)
		self.vars = []
		for pick in picks:
			var = IntVar()
			chk = Checkbutton(self, text=pick, variable=var)
			chk.pack(side=side, anchor=anchor, expand=YES)
			self.vars.append(var)

	def state(self):
		return map((lambda var: var.get()), self.vars)

class eye_circle:
	def __init__(self,master = None):
		self.x=0
		self.y=0
		self.canvas = Canvas(master)
		self.circle = self.canvas.create_oval(0,0,10,10)
		self.canvas.pack()
		self.movement()
		# while True:
		# 	self.movement()

	def movement(self):
		self.canvas.move(self.circle,self.x,self.y)



if __name__ == '__main__':

	root = Tk()
	root.geometry("400x400")
	pupil_circle = eye_circle(root)
	# root.bind("move",lambda  e: eye_circle.movement(e))
	experiment_variables = Checkbar(root, ['log', 'pupil'])
	experiment_variables.pack(side=TOP, fill=X)
	experiment_variables.config(relief=GROOVE, bd=2)



	def startZMQ():
		zmq_thread = ZMQ_listener(name='ZMQ_listener', args=[True])
		zmq_thread.start()
	def start_UDP_listener():
		udp_listener_thread = UDP_listener(args=[])
		udp_listener_thread.start()

	def allstates():
		print(list(experiment_variables.state()))


	def allstop():
		root.destroy()
		sys.exit()


	# del(zmq_thread)
	# ZMQ_listener_main().stop()
	# zmq_thread.join()
	Button(root,text = 'UDP listener start', command = start_UDP_listener).pack(side=TOP)
	Button(root, text='Pupil start',command=startZMQ).pack(side=TOP)
	Button(root, text='Peek', command=allstates).pack(side=RIGHT)
	Button(root, text='Quit', command=allstop).pack(side=BOTTOM)

	root.mainloop()
