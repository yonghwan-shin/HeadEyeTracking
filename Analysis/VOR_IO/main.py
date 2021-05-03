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
A way to call dataset in one trial.
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

"""
Basic methods to visualize dataset
"""
plt.plot(holo.timestamp, holo.head_rotation_y, label='head yaw')
plt.plot(eye.timestamp, eye.phi, label='eye yaw')
plt.plot(holo.timestamp, holo.head_rotation_y + eye.phi, label='head + eye')
plt.plot(holo.timestamp, holo.Phi, label='target center position', linestyle='--')
plt.axvline(initial_contact_time)
plt.legend()
plt.title("Horizontal Movements")
plt.show()

plt.plot(holo.timestamp, holo.head_rotation_x, label='head pitch')
plt.plot(eye.timestamp, -eye.theta, label='eye pitch')  # Note that eye.theta has opposite direction
plt.plot(holo.timestamp, holo.head_rotation_x - eye.theta, label='head + eye')
plt.plot(holo.timestamp, holo.Theta, label='target center position', linestyle='--')
plt.plot(holo.timestamp,eased_rotation,label = 'easing function')
plt.axvline(initial_contact_time)
plt.legend()
plt.title("Vertical Movements")
plt.show()


"""
Sample algorithm
"""
slow = 0.05
fast = 0.5
r = 0.5
cursor_x=[]
for i in range(len(holo.timestamp)):
    if i == 0:
        cursor_x.append(holo.head_rotation_y[0])
    else:
        if holo.timestamp[i] < initial_contact_time:
            r = fast
        else:
            r = slow
        previous_cursor = cursor_x[i - 1]
        eye_cursor = holo.head_rotation_y[i] + eye.phi[i]  # estimation
        estimated_direction = eye_cursor - holo.head_rotation_y[i]
        # head_movement = holo.head_rotation_y[i] - holo.head_rotation_y[i - 1]  # head movement
        head_movement = holo.head_rotation_y.diff(1)[i - 10:i].mean()
        correct_direction = True if head_movement * estimated_direction > 0 else False
        if correct_direction:  # head is moving towards
            if abs(estimated_direction) < abs(eye_cursor - previous_cursor):  # if actual Head is closer than cursor
                r = fast
            else:
                r = slow
        else:
            r = slow
        # new_cursor = lerp_one_frame(previous_cursor, holo.head_rotation_y[i],
        #                             holo.timestamp[i] - holo.timestamp[i - 1], r)
        new_cursor = previous_cursor * (1-r) + r* holo.head_rotation_y[i]
        cursor_x.append(new_cursor)

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
