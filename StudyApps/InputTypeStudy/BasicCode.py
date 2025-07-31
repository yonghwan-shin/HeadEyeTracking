#%% IMPORTS
from BasicFileHandling import *
import matplotlib.pyplot as plt

#%% SEE WHAT DATA IS LIKE FOR A ONE TRIAL
### There are 5 repetitions, you can change the variable rep from 0 to 4  ###
rep = 0
d, success_record = read_data(11, rep, 'Eye', "Dwell", "Walk")
### For each repetition, there are 9 trials, you can change the variable t from 0 to 8 ###
t=7
data = split_target(d)
### temp_data is a DataFrame for the t-th trial ###
temp_data = data[t].reset_index()
temp_data.timestamp -= temp_data.timestamp.values[0]
#### a simple plot of the cursor angular distance over time ####
plt.plot(temp_data.timestamp.values, temp_data.cursor_angular_distance.values, label='Head Horizontal')
plt.axhline(3, color='red', linestyle='--', label='Threshold')
plt.title(success_record[t])
plt.show()