from subjectHandling import *


def filtering(pupil_data_file):  # change function name
	[pupil_timestamp, pupil_norm_pos_x, pupil_norm_pos_y, pupil_confidence] = organise_pupil_data(pupil_data_file)
