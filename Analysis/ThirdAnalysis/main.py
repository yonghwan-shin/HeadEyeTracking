from filehandling import *
import matplotlib.pyplot as plt

data_sample = read_hololens_json(3, 'W', 1, 9991)

plt.plot(data_sample.angular_distance)
plt.show()