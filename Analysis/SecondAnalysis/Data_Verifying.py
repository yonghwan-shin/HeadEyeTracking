from pathlib import Path
from Analysing_functions import *
import itertools
import seaborn as sns
from scipy import interpolate
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics

subjects = range(201, 212)
targets = range(8)
envs = ['U', 'W']
poss = ['W', 'S']
# blocks = range(1, 5)
blocks = range(5)
dataset_folder_name = "2ndData"
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT.parent.parent / 'Datasets' / dataset_folder_name
EYE_DATA_PATH = DATA_ROOT / 'refined_eye_data'

print('DATA ROOT PATH:', DATA_ROOT)
# %%
def make_pkl():
    # info: target,env,pos,block
    filename = lambda info: 'T' + str(info[0]) + "_E" + str(info[1]) + '_P' + str(info[2]) + '_B' + str(info[3])
    # hololens_subject_folder = lambda subject: DATA_ROOT/'hololens_data' / ('compressed_sub' + str(subject))
    hololens_subject_folder = DATA_ROOT / 'hololens_data'
    whole_data = pd.DataFrame(columns=['subject', 'target', 'env', 'pos', 'block', 'eye', 'holo', 'mark'])
    for subject in subjects:
        for target, env, pos, block in itertools.product(targets, envs, poss, blocks):
            mark = ''
            eye_file = EYE_DATA_PATH.rglob('*' + filename([target, env, pos, block]) + "*.csv")
            holo_file_path = (hololens_subject_folder / ('compressed_sub' + str(subject)))
            holo_file = holo_file_path.rglob("*" + filename([target, env, pos, block]) + "*.csv")
            for file in eye_file:   eye_file = file
            for file1 in holo_file: holo_file = file1
            try:
                if eye_file.exists() and eye_file.is_file():
                    eye_dataframe = pd.read_csv(eye_file, index_col=False)
                if holo_file.exists() and holo_file.is_file():
                    holo_dataframe = pd.read_csv(holo_file, index_col=False, header=1)
            except:
                raise IOError("error in reading eye,holo file", eye_file)

            eye_dataframe.update({'timestamp': eye_dataframe['timestamp'] - eye_dataframe.head(1)['timestamp'].values[0]})
            holo_dataframe.update(
                {'Timestamp': holo_dataframe['Timestamp'] - holo_dataframe.head(1)['Timestamp'].values[0]})
            if eye_dataframe.shape[0] < 600:
                mark = 'short_eye,'
                print('too short eye data')
            else:
                refined_eye_dataframe = eye_dataframe[eye_dataframe['confidence'] > 0.6]
                refined_eye_dataframe = refined_eye_dataframe.drop(
                    columns=['circle_3d', 'ellipse', 'location', 'diameter', 'sphere', 'projected_sphere',
                             'model_confidence', 'model_id', 'model_birth_timestamp', 'topic', 'id', 'method'])
                if refined_eye_dataframe.shape[0] < 400:
                    mark = mark + 'short_confidence,'
                    print('too short confident data')

            if holo_dataframe.shape[0] < 300:
                mark = mark + 'short_holo,'
                print('too short holo data')
            head_position_start = float(holo_dataframe.head(1)['HeadPositionZ'])
            head_position_end = float(holo_dataframe.tail(1)['HeadPositionZ'])
            if pos == 'W' and (head_position_end - head_position_start) < 4.5:
                mark = mark + 'short_walk'
                print('Short walk length:', (head_position_end - head_position_start))
            whole_data = whole_data.append({
                # 'info': [subject, target, env, pos, block],
                'subject': str(subject),
                'target': str(target),
                'env': str(env),
                'pos': str(pos),
                'block': str(block),

                'eye': [refined_eye_dataframe] if refined_eye_dataframe is not None else [eye_dataframe],
                'holo': [holo_dataframe],
                'mark': [mark]

            }, ignore_index=True)

# %% save
# whole_data.to_pickle("whole_data.pkl")
# %% test
# errors = whole_data[whole_data['data']['mark'].str.contains('short')]
# print('error count:', errors.shape[0])


# %% manual filtering



def manual_filtering(subject, target, env, pos, block):
    df = whole_data.loc[(whole_data['subject'] == subject) &
                        (whole_data['target'] == target) &
                        (whole_data['env'] == env) &
                        (whole_data['pos'] == pos) &
                        (whole_data['block'] == block)
                        ]
    # print(df)
    # print(df)
    # data=df['data']
    # print(data)
    eye = df['eye'].values[0][0]
    holo = df['holo'].values[0][0]
    mark = df['mark'].values[0][0]
    return eye, holo, mark


# %%
# eye,holo,mark =\
whole_data = pd.read_pickle('whole_data.pkl')
for subject, target, env, pos, block in itertools.product(subjects, targets, envs, poss, blocks):
    eye, holo, mark = manual_filtering(str(subject), str(target), str(env), str(pos), str(block))
    # holo.update({'Timestamp': holo_dataframe['Timestamp'] + 0.5})
    filtered_x = butterworth_filter(eye['norm_x'])
    rolling_x = rolling_filter(eye['norm_x'])
    plt.plot(eye['norm_x'])
    # plt.plot(filtered_x)
    plt.plot(rolling_x)
    plt.show()

    pass
