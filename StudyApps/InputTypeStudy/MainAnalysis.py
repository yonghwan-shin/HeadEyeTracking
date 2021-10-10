"""
cursor_angular_distance', 'start_num', 'end_num', 'timestamp',
'target_position_x', 'target_position_y', 'target_position_z',
'head_position_x', 'head_position_y', 'head_position_z',
'head_rotation_x', 'head_rotation_y', 'head_rotation_z',
'head_forward_x', 'head_forward_y', 'head_forward_z', 'cursor_type',
'target_name', 'origin_x', 'origin_y', 'origin_z', 'direction_x',
'direction_y', 'direction_z', 'ray_origin_x', 'ray_origin_y',
'ray_origin_z', 'ray_direction_x', 'ray_direction_y',
'ray_direction_z']
"""
# %%
from FileHandling import *
import matplotlib.pyplot as plt

pd.set_option('mode.chained_assignment', None)  # <==== 경고를 끈다
# %%
# cursorTypes= ['HEAD','EYE','HAND']
draw_plot= False
cursorTypes = ['EYE']
sub_num =4
summary = pd.DataFrame(columns=['subject_num','posture','cursor_type','repetition','target_num'])
for cursor_type in cursorTypes:
    for rep in range(5):
        data = read_hololens_data(sub_num, 'STAND', cursor_type, rep)
        walkdata = read_hololens_data(sub_num, 'WALK', cursor_type, rep)
        # print(data.columns)
        splited_data = split_target(data)
        walk_splited_data = split_target(walkdata)

        stand_offset_means=[]
        stand_offset_stds = []
        walk_offset_means=[]
        walk_offset_stds = []

        for i in range(9):
            temp = splited_data[i]
            walk_temp = walk_splited_data[i]
            temp.reset_index(inplace=True)
            temp.timestamp -= temp.timestamp.values[0]
            walk_temp.reset_index(inplace=True)
            walk_temp.timestamp -= walk_temp.timestamp.values[0]
            initial_contact_time = temp[temp.target_name == "Target_"+str(i)].timestamp.values[0]
            walk_initial_contact_time = walk_temp[walk_temp.target_name == "Target_"+str(i)].timestamp.values[0]

            dwell_temp = temp[temp.timestamp > initial_contact_time]
            offset_mean = dwell_temp['cursor_angular_distance'].mean()
            offset_std = dwell_temp['cursor_angular_distance'].std()
            walk_dwell_temp = walk_temp[walk_temp.timestamp > walk_initial_contact_time]
            walk_offset_mean = walk_dwell_temp['cursor_angular_distance'].mean()
            walk_offset_std = walk_dwell_temp['cursor_angular_distance'].std()
            stand_offset_means.append(offset_mean)
            stand_offset_stds.append(offset_std)
            walk_offset_means.append(walk_offset_mean)
            walk_offset_stds.append(walk_offset_std)
            # print(i,'stand',offset_mean,offset_std)
            # print(i, 'walk', walk_offset_mean, walk_offset_std)
            # print("")

            if(draw_plot):
                plt.plot(temp.timestamp, temp.cursor_angular_distance, label='stand')
                plt.plot(walk_temp.timestamp, walk_temp.cursor_angular_distance, label='walk')
                plt.axvline(initial_contact_time)
                plt.axvline(walk_initial_contact_time)
                # plt.text(initial_contact_time,0,'initial contact time',)
                plt.legend()
                plt.title(f'{cursor_type} angular offset from :target' + str(i))
                plt.show()
        print(cursor_type,rep,'\n','stand offset mean',sum(stand_offset_means)/len(stand_offset_means),
              'stand offset std',sum(stand_offset_stds)/len(stand_offset_stds),'\n',
              'walk offset mean', sum(walk_offset_means) / len(walk_offset_means),
              'walk offset std', sum(walk_offset_stds) / len(walk_offset_stds)
              )
