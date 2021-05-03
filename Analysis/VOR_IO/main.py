# %%
import pandas as pd
from AnalysingFunctions import *
from FileHandling import *
import time
"""
NOTE: vertical/horizontal parameters
Vertical:     imu-x/head-rotation-x/holo-the/eye-y/eye-the
Horizontal:   imu-z/head-rotation-y/holo-phi/eye-x/eye-phi
"""


""" 
A way to call all dataset in one trial.
studynum 2 : comparing stand / walking
studynum 3: walking only
note that returned 'initial_contact_time' is the first timestamp when cursor gets into the target.
"""
subject = 3
env = 'U'
target = 5
block = 4
holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
                                                       subject=subject, study_num=3)
"""
Remove low-confidence pupil data. If there is another way to filter out the outlier, comment this out.
"""
eye = eye[eye.confidence > 0.8]

"""
make three dataset into same framerate by interpolation. Original framerate is; Holo (60Hz), eye (120Hz), imu (200Hz)
"""
holo = interpolate_dataframe(holo, framerate=60)
eye = interpolate_dataframe(eye, framerate=60)
imu = interpolate_dataframe(imu, framerate=60)

"""
Easing functions
"""
# simplest method
eased_rotation = easing_linear(holo.head_rotation_x, coef=0.05)


# %%
"""
To iterate along whole trials,
"""
# subjects = range(1, 17)
# envs = ['U', 'W']
# targets = range(8)
# blocks = range(1, 5)
#
# for subject, env, target, block in itertools.product(
#         subjects, envs, targets, blocks
# ):
#     holo, eye, imu, initial_contact_time = bring_one_trial(target=target, env=env, posture='W', block=block,
#                                                            subject=subject, study_num=3)
#     eye = eye[eye.confidence > 0.8]
#     holo = interpolate_dataframe(holo)
#     eye = interpolate_dataframe(eye)
#     imu = interpolate_dataframe(imu)
