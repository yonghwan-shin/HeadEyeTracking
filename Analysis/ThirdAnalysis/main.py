from filehandling import *
import matplotlib.pyplot as plt

data_sample = read_hololens_json(3, 'W', 1, 9991)


# plt.plot(data_sample.angular_distance)
# plt.show()
# a = data_sample.head_position.apply(pd.Series)
# data_sample['xx'] = a['x']



# data_sample['head_position'+'_x'] = data_sample['head_position'].apply(pd.Series)['x']
data = refining_hololens_dataframe(data_sample)
