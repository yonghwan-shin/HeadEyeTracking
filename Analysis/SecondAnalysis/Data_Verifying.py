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
                    for j in range(-12, 12):
                        if i+j not in refined_eye_dataframe.index:
                            continue
                        refined_eye_dataframe.at[i + j, 'drop'] = 'drop'
                print(refined_eye_dataframe.shape[0]," ->" ,end="")
                refined_eye_dataframe.drop(refined_eye_dataframe[refined_eye_dataframe['drop'] == 'drop'].index,
                                           inplace=True)
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

            whole_data = whole_data.append(
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
    # output_eye = whole_data[['subject','target','env','pos','block','eye']]
    # output_holo = whole_data[['subject', 'target', 'env', 'pos', 'block', 'holo']]
    # output_mark = whole_data[['subject', 'target', 'env', 'pos', 'block', 'mark']]
    # output_eye.to_pickle('whole_eye.pkl.gz')
    # output_holo.to_pickle('whole_holo.pkl.gz')
    # output_mark.to_pickle('whole_mark.pkl.gz')
    whole_data.to_pickle("whole_data3.pkl")
    return whole_data


# %% manual filtering
def manual_filtering(whole_data, subject, target, env, pos, block):
    df = whole_data.loc[
        (whole_data["subject"] == subject)
        & (whole_data["target"] == target)
        & (whole_data["env"] == env)
        & (whole_data["pos"] == pos)
        & (whole_data["block"] == block)
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


# %%
if __name__ == "__main__":
    analysis()
    # make_pkl()
