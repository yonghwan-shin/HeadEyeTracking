IMU/Pupil(eye-tracker) data are on {subject-num}/IMU* or EYE*
Pupil data have form of timestamp, pupil-json with total row length on header. (ignore 'norm_x', 'norm_y' on header)
Its json rows have unknown type-error while decompiling in Python. It works with the 'demjson' package.

IMU data have form of timestamp, quaternion values, with total length on header.

HMD-experiment data are on {subject-num}_holo/
Its file has form of full json file : {'filename','subject number', 'data'}
data : {timestamp, head-position/rotation/forward vector, target-position, angle-difference between head forward vector and head to target vector} 
