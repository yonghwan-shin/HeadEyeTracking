import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z



# euler_Original = (np.random.random(3)).tolist()
# # Generate random rotation angles for XYZ within the range [0, 360)
#
# quat = euler_to_quaternion(euler_Original[0], euler_Original[1], euler_Original[2])
# # Convert to Quaternion
# newEulerRot = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])
# # Convert the Quaternion to Euler angles
#
# print(euler_Original)
# print(np.degrees(euler_Original[0]),np.degrees(euler_Original[1]),np.degrees(euler_Original[2]))
# print(newEulerRot)

# scipy rotation functions test...
##########################
# r = R.from_euler('zxy', [60, 40, 50], degrees=True)
# print(r.as_euler('zxy', degrees=True))
# print(r.as_quat())
# rr = R.from_quat(r.as_quat())
# print(rr.as_euler('zxy', degrees=True))
###########################