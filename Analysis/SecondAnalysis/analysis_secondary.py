#%%
from new import *
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff_freq, fs, order=3, real_time=False):
    nyq_freq = fs / 2
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    if real_time == False:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


# subjects = range(201, 212)
# poss = ['W','S']
poss= ['S']
# envs = ['U','W']
envs=['U']
# targets=range(8)
targets=[2]
blocks=[3]
# blocks=range(5)
# blocks = [1,2,3,4]
subjects= [201]
# poss=['W']
# envs=['W']
# targets=[0]
# blocks=[0]
output=pd.DataFrame()
# TODO : summary save for each subject
for subject in subjects:
    output=pd.DataFrame()
    [imu_file_list, eye_file_list, hololens_file_list] = FileHandling.get_one_subject_files(subject, refined=True)
    for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
        current_info = [target, env, pos, block]
        try:
            print('analysing',subject,current_info)

            holo = FileHandling.file_as_pandas(FileHandling.get_file_by_info(hololens_file_list, current_info))
            eye = FileHandling.file_as_pandas(FileHandling.get_file_by_info(eye_file_list, current_info),
                                              refined=True)
            eye.timestamp = eye.timestamp - eye.timestamp[0]
            eye=eye[eye.confidence>0.9]
            holo.Timestamp  = holo.Timestamp - holo.Timestamp[0]
            holo.HeadRotationY = butter_lowpass_filter(holo.HeadRotationY,5,60)

            eye['filtered_norm_x'] = butter_lowpass_filter(eye.norm_x,5,120)


            plt.plot(holo.Timestamp,holo.HeadRotationY)
            # plt.show()
            plt.plot(eye.timestamp,(eye.filtered_norm_x-eye.filtered_norm_x.mean())*200)
            plt.show()
            plt.plot(holo.Timestamp, holo.HeadRotationY.diff(1)*60)
            plt.show()
        except Exception as e:
            print(e)