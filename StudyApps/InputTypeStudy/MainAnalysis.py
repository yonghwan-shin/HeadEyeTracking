#%%
from FileHandling import *
import matplotlib.pyplot as plt
data = read_hololens_data(0,'STAND','HEAD',0)
# print(data.columns)
splited_data = split_target(data)
temp = splited_data[0]

temp.cursor_angular_distance.plot();plt.show()