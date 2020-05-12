# %%
from pathlib import Path
from Analysing_functions import *
import itertools
import seaborn as sns
from scipy import interpolate
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

subjects = range(201, 212)
targets = range(8)
envs = ["U", "W"]
# poss=['W']
poss = ["W", "S"]
# blocks = range(1, 5)
blocks = range(5)
dataset_folder_name = "2ndData"
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT.parent.parent / "Datasets" / dataset_folder_name
EYE_DATA_PATH = DATA_ROOT / "refined_eye_data"

print("DATA ROOT PATH:", DATA_ROOT)


# %%
def make_pkl():
    # info: target,env,pos,block
    filename = (
        lambda info: "T"
                     + str(info[0])
                     + "_E"
                     + str(info[1])
                     + "_P"
                     + str(info[2])
                     + "_B"
                     + str(info[3])
    )
    # hololens_subject_folder = lambda subject: DATA_ROOT/'hololens_data' / ('compressed_sub' + str(subject))
    hololens_subject_folder = DATA_ROOT / "hololens_data"
    whole_data = pd.DataFrame(
        columns=["subject", "target", "env", "pos", "block", "eye", "holo", "mark"]
    )
    for subject in subjects:
        subject_data = pd.DataFrame(
            columns=["subject", "target", "env", "pos", "block", "eye", "holo", "mark"]
        )
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            print(subject, target, env, pos, block)
            mark = ""
            eye_file = EYE_DATA_PATH.rglob(
                "*" + filename([target, env, pos, block]) + "*.csv"
            )
            holo_file_path = hololens_subject_folder / ("compressed_sub" + str(subject))
            holo_file = holo_file_path.rglob(
                "*" + filename([target, env, pos, block]) + "*.csv"
            )
            for file in eye_file:
                eye_file = file
            for file1 in holo_file:
                holo_file = file1
            try:
                if eye_file.exists() and eye_file.is_file():
                    eye_dataframe = pd.read_csv(eye_file, index_col=False)
                if holo_file.exists() and holo_file.is_file():
                    holo_dataframe = pd.read_csv(holo_file, index_col=False, header=1)
            except:
                raise IOError("error in reading eye,holo file", eye_file)

            eye_dataframe.update(
                {
                    "timestamp": eye_dataframe["timestamp"]
                                 - eye_dataframe.head(1)["timestamp"].values[0]
                }
            )
            holo_dataframe.update(
                {
                    "Timestamp": holo_dataframe["Timestamp"]
                                 - holo_dataframe.head(1)["Timestamp"].values[0]
                }
            )
            if eye_dataframe.shape[0] < 600:
                mark = "short_eye,"
                print("too short eye data")
            else:

                # refined_eye_dataframe = eye_dataframe[eye_dataframe["confidence"] > 0.6]
                refined_eye_dataframe = eye_dataframe.copy()
                refined_eye_dataframe['drop'] = ""
                lows = np.array([])
                for index, row in refined_eye_dataframe.iterrows():
                    if refined_eye_dataframe.loc[index, 'confidence'] < 0.6:
                        lows = np.append(lows, index)
                for i in lows:
                    for j in range(-6, 6):
                        if i + j not in refined_eye_dataframe.index:
                            continue
                        refined_eye_dataframe.at[i + j, 'drop'] = 'drop'
                print(refined_eye_dataframe.shape[0], " ->", end="")
                # refined_eye_dataframe.drop(refined_eye_dataframe[refined_eye_dataframe['drop'] == 'drop'].index,
                #                            inplace=True)
                print(refined_eye_dataframe.shape[0])

                refined_eye_dataframe = refined_eye_dataframe.drop(
                    columns=[
                        "circle_3d",
                        "ellipse",
                        "location",
                        "diameter",
                        "sphere",
                        "projected_sphere",
                        "model_confidence",
                        "model_id",
                        "model_birth_timestamp",
                        "topic",
                        "id",
                        "method",
                    ]
                )
                if refined_eye_dataframe.shape[0] < 400:
                    mark = mark + "short_confidence,"
                    print("too short confident data")

            if holo_dataframe.shape[0] < 300:
                mark = mark + "short_holo,"
                print("too short holo data")
            head_position_start = float(holo_dataframe.head(1)["HeadPositionZ"])
            head_position_end = float(holo_dataframe.tail(1)["HeadPositionZ"])
            if pos == "W" and (head_position_end - head_position_start) < 4.5:
                mark = mark + "short_walk"
                print("Short walk length:", (head_position_end - head_position_start))
            rs = []
            thetas = []
            phis = []
            for index, row in holo_dataframe.iterrows():
                x = row['TargetPositionX'] - row['HeadPositionX']
                y = row['TargetPositionY'] - row['HeadPositionY']
                z = row['TargetPositionZ'] - row['HeadPositionZ']
                [r, theta, phi] = asSpherical([x, z, y])
                rs.append(r)
                thetas.append(90 - theta)
                phis.append(90 - phi)

            holo_dataframe['R'] = rs
            holo_dataframe['Theta'] = thetas
            holo_dataframe['Phi'] = phis

            subject_data = subject_data.append(
                {
                    # 'info': [subject, target, env, pos, block],
                    "subject": str(subject),
                    "target": str(target),
                    "env": str(env),
                    "pos": str(pos),
                    "block": str(block),
                    "eye": [refined_eye_dataframe]
                    if refined_eye_dataframe is not None
                    else [eye_dataframe],
                    "holo": [holo_dataframe],
                    "mark": [mark],
                },
                ignore_index=True,
            )
        subject_data.to_pickle('subject_'+str(subject)+'.pkl')
    # output_eye = whole_data[['subject','target','env','pos','block','eye']]
    # output_holo = whole_data[['subject', 'target', 'env', 'pos', 'block', 'holo']]
    # output_mark = whole_data[['subject', 'target', 'env', 'pos', 'block', 'mark']]
    # output_eye.to_pickle('whole_eye.pkl.gz')
    # output_holo.to_pickle('whole_holo.pkl.gz')
    # output_mark.to_pickle('whole_mark.pkl.gz')
    # whole_data.to_pickle("whole_data3.pkl")
    return whole_data


# %% manual filtering
def manual_filtering(whole_data, subject, target, env, pos, block):
    df = whole_data.loc[
        (whole_data["subject"] == str(subject))
        & (whole_data["target"] == str(target))
        & (whole_data["env"] == env)
        & (whole_data["pos"] == pos)
        & (whole_data["block"] == str(block))
        ]
    # print(df)
    # print(df)
    # data=df['data']
    # print(data)
    eye = df["eye"].values[0][0]
    holo = df["holo"].values[0][0]
    mark = df["mark"].values[0][0]
    return eye, holo, mark


# %% save
# whole_data = make_pkl()
# whole_data.to_pickle("whole_data.pkl")
# %% test
# errors = whole_data[whole_data['data']['mark'].str.contains('short')]
# print('error count:', errors.shape[0])

# %%
def analysis():
    # eye,holo,mark =\
    whole_data = pd.read_pickle("whole_data3.pkl")
    # plt.ion()

    for subject, target, env, pos, block in itertools.product(
            subjects, targets, envs, poss, blocks
    ):
        eye, holo, mark = manual_filtering(whole_data,
                                           str(subject), str(target), str(env), str(pos), str(block)
                                           )
        fig, axs = plt.subplots(2, 1, figsize=[8, 8], sharex=True)
        axs[0].set_title(str(subject) + str(target) + str(env) + str(pos) + str(block))
        holo.update({'Timestamp': holo['Timestamp'] + 0.05})
        eye.update({'python_timestamp': (eye['python_timestamp'] - eye['python_timestamp'].head(1).values[0])})
        # rolling_x = rolling_filter(eye['norm_x'])
        # fixation, saccade = saccade_filter(0.005, eye['norm_x'])
        # outliers = find_outlier(eye["norm_x"], threshold=5)
        # eye.drop(eye.index[outliers], inplace=True)
        # low_densities, temps = find_densi ty(eye['timestamp'].to_numpy(), threshold=0.8)

        intp_holoY = interpolate.interp1d(holo["Timestamp"], holo["HeadRotationY"])
        # intp_holoY = interpolate.splrep(holo["Timestamp"], holo["HeadRotationY"])
        intp_holoPhi = interpolate.interp1d(holo['Timestamp'], holo['Phi'])

        timestamps = np.arange(1, 5.5, 1 / 120)

        if eye["timestamp"].tail(1).values[0] < 5.5 or eye["timestamp"].head(1).values[0] > 1:
            print('error in eye timestamp')
            continue;

        intp_eye_x = interpolate.interp1d(eye["timestamp"], eye["norm_x"])
        py_intp_eye_x = interpolate.interp1d(eye["python_timestamp"], eye["norm_x"])
        # for section in temps:
        #     eye_section = eye.iloc[section, :]
        #     if eye_section['timestamp'].head(1).values[0] < 0.05:
        #         start = 0.05
        #     else:
        #         start = eye_section['timestamp'].head(1).values[0]
        #     timestamps = np.arange(start,eye_section['timestamp'].tail(1).values[0], 1 / 120)
        #
        #     eye_x = intp_eye_x(timestamps)
        #     if len(eye_x)<20:
        #         continue
        #     filtered_x = butterworth_filter(eye_x, fc=5)
        #     target_x = intp_holoY(timestamps) - intp_holoPhi(timestamps)
        #
        #     axs[0].plot(timestamps, filtered_x)
        #     axs[0].scatter(eye['timestamp'], eye['norm_x'], marker='x', alpha=0.5)
        #     axs[1].plot(timestamps, target_x)
        #     slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
        #         filtered_x, target_x)
        #     axs[1].plot(timestamps, filtered_x * slope_vertical + intercept_vertical, 'r')
        #     pass

        eye_x = intp_eye_x(timestamps)
        py_eye_x = py_intp_eye_x(timestamps)
        filtered_x = butterworth_filter(eye_x, fc=5)
        py_filtered_x = butterworth_filter(py_eye_x, fc=5)

        holo_y = intp_holoY(timestamps)
        holo_phi = intp_holoPhi(timestamps)

        axs[0].plot(timestamps, filtered_x)
        axs[0].plot(timestamps, py_filtered_x)

        plt.show()
        # axs[0].scatter(eye['timestamp'], eye['norm_x'], marker='x', alpha=0.5)

        # axs[1].plot(timestamps, holo_y - holo_phi)
        # norm_eye_x = normalize(filtered_x,(holo_y-holo_phi))
        # # axs[1].plot(timestamps,norm_eye_x)
        # # axs[1].plot(timestamps,norm_eye_x+holo_y-holo_phi,'r')
        # axs[1].hlines([0], xmax=timestamps[-1], xmin=timestamps[0])
        slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
            filtered_x, holo_y - holo_phi)
        # axs[1].plot(timestamps,filtered_x*slope_vertical+intercept_vertical,'r')

        # axs[0].scatter(eye_x, holo_y - holo_phi)
        # axs[0].plot(eye_x, eye_x * slope_vertical + intercept_vertical)

    plt.grid(True)
    plt.show()

    pass
def one_subject_analysis(subject_num):
    step = 1/200
    sns.set()


    whole_eye_vertical = np.array([])
    whole_eye_horizontal = np.array([])
    whole_dff_vertical = np.array([])
    whole_dff_horizontal = np.array([])
    whole_target_vertical = np.array([])
    whole_target_horizontal = np.array([])
    whole_dff_norm_vertical = np.array([])
    whole_dff_norm_horizontal = np.array([])
    data = pd.read_pickle('subject_'+str(subject_num)+'.pkl')
    for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
        eye,holo,mark =manual_filtering(data,subject_num,target,env,pos,block)
        if mark is None:
            print('something wrong,,,', mark)
            continue
        else:
            low_start = []
            low_end = []
            for index, row in eye.iterrows():
                if index == 0:
                    # continue
                    if eye.loc[index, 'drop'] == 'drop':
                        low_start.append(eye.loc[index, 'timestamp'])
                else:
                    if eye.loc[index, 'drop'] == 'drop' and eye.loc[index - 1, 'drop'] != 'drop':
                        low_start.append(eye.loc[index, 'timestamp'])
                    if eye.loc[index, 'drop'] != 'drop' and eye.loc[index - 1, 'drop'] == 'drop':
                        low_end.append(eye.loc[index, 'timestamp'])

        holo.update({'Timestamp': holo['Timestamp'] + 0.05})
        eye.update({'timestamp': (eye['timestamp'] - eye['timestamp'].head(1).values[0])})
        intp_holoX = interpolate.interp1d(holo['Timestamp'],holo['HeadRotationX'])
        intp_holoY = interpolate.interp1d(holo["Timestamp"], holo["HeadRotationY"])
        intp_holoPhi = interpolate.interp1d(holo['Timestamp'], holo['Phi'])
        intp_holoThe = interpolate.interp1d(holo['Timestamp'],holo["Theta"])
        intp_eyeX = interpolate.interp1d(eye['timestamp'],eye['norm_x'])
        intp_eyeY = interpolate.interp1d(eye['timestamp'],eye['norm_y'])
        target_verticals = np.array([])
        target_horizontals = np.array([])
        eye_linregress_verticals = np.array([])
        eye_linregress_horizontals = np.array([])
        eye_filtered_verticals = np.array([])
        eye_filtered_horizontals = np.array([])
        print('start',low_start)
        print('end',low_end)
        timeline = parse_timeline(np.asarray(low_start),np.asarray(low_end))
        test_axis = np.arange(0.9,6.1,step)
        times = np.array([])
        fig, axs = plt.subplots(2, 1, figsize=[6, 12], sharex=True)
        for time in timeline:
            t1 = np.where((test_axis>time[0])&(test_axis<time[1]))
            t = np.arange(math.ceil(time[0]*100)/100,math.floor(time[1]*100)/100,step)
            if len(t1[0])<50: continue
            times = np.append(times,t)
            print(time)
            target_vertical = intp_holoX(t) + intp_holoThe(t)
            target_horizontal = intp_holoY(t) - intp_holoPhi(t)
            eye_vertical = intp_eyeY(t)
            eye_horizontal = intp_eyeX(t)
            filtered_eye_vertical = one_euro(eye_vertical,freq=120)
            filtered_eye_horizontal = one_euro(eye_horizontal,freq=120)
            filtered_target_vertical = one_euro(target_vertical,freq=120)
            filtered_target_horizontal = one_euro(target_horizontal,freq=120)
            slope_vertical, intercept_vertical, r_vertical, p_vertical, std_err_vertical = stats.linregress(
                filtered_eye_vertical, filtered_target_vertical)
            slope_horizontal, intercept_horizontal, r_horizontal, p_horizontal, std_err_horizontal = stats.linregress(
                filtered_eye_horizontal, filtered_target_horizontal)

            if p_vertical > 0.05 or p_horizontal > 0.05:
                continue
            eye_linregress_vertical = slope_vertical * filtered_eye_vertical + intercept_vertical
            eye_linregress_horizontal = slope_horizontal * filtered_eye_horizontal + intercept_horizontal
            eye_normalised_vertical = normalize(eye_vertical, target_vertical)

            target_verticals = np.append(target_verticals, filtered_target_vertical)
            target_horizontals = np.append(target_horizontals, filtered_target_horizontal)

            eye_linregress_verticals = np.append(eye_linregress_verticals, eye_linregress_vertical)
            eye_linregress_horizontals = np.append(eye_linregress_horizontals, eye_linregress_horizontal)
            eye_filtered_verticals = np.append(eye_filtered_verticals, filtered_eye_vertical)
            eye_filtered_horizontals = np.append(eye_filtered_horizontals, filtered_eye_horizontal)
            axs[0].plot(t,filtered_target_vertical,color='black')
            axs[0].plot(t,eye_linregress_vertical,color='blue')
            axs[1].plot(t,filtered_target_horizontal,color='black')
            axs[1].plot(t,eye_linregress_horizontal,color='blue')
        plt.show()
        whole_dff_vertical = np.append(whole_dff_vertical, target_verticals - eye_linregress_verticals)
        whole_dff_horizontal = np.append(whole_dff_horizontal, target_horizontals - eye_linregress_horizontals)
        whole_target_vertical = np.append(whole_target_vertical, target_verticals)
        whole_target_horizontal = np.append(whole_target_horizontal, target_horizontals)
        whole_dff_norm_vertical = np.append(whole_dff_norm_vertical, eye_normalised_vertical)
        whole_dff_norm_horizontal = np.append(whole_dff_norm_horizontal, eye_normalised_vertical)
        whole_eye_vertical = np.append(whole_eye_vertical, eye_filtered_verticals)
        whole_eye_horizontal = np.append(whole_eye_horizontal, eye_filtered_horizontals)
    fig, axs = plt.subplots(2, 1, figsize=[6, 12], sharex=True)
    sns.scatterplot(whole_target_horizontal, whole_target_vertical, ax=axs[0], markers='+', alpha=0.3)
    confidence_ellipse(whole_target_horizontal, whole_target_vertical, n_std=3, edgecolor='firebrick', ax=axs[0])
    sns.scatterplot(whole_dff_horizontal, whole_dff_vertical, ax=axs[1], markers='+', alpha=0.3)
    confidence_ellipse(whole_dff_horizontal, whole_dff_vertical, n_std=3, edgecolor='fuchsia', ax=axs[1])
    confidence_ellipse(whole_target_horizontal, whole_target_vertical, n_std=3, edgecolor='firebrick', ax=axs[1])
    axs[0].axis('equal')
    axs[1].axis('equal')
    axs[0].set_title('target')
    axs[1].set_title('compensated-linregress')
    plt.legend()
    fig.suptitle('' + str(subject_num), horizontalalignment='left', verticalalignment='top', fontsize=15)
    plt.savefig('target_plots_' + str(subject_num - 200) + '.pdf')
    # fig.tight_layout()
    plt.show()
    print('drawing plot',subject_num)
# %%
if __name__ == "__main__":
    one_subject_analysis(201)

    # analysis()
    # make_pkl()
