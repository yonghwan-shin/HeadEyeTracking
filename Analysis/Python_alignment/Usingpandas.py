from subjectHandling import *
from oneEuroFilter import *
import itertools

# Full test
TARGETS = range(8)
ENVIRONMENTS = ['U', 'W']
POSTURES = ['W', 'S']
BLOCKS = range(5)


def analyse_refined_data():
	# get_refined_files('imu_1_T0_EU_PS_B0.csv')
	for subjectNum in range(1, 17):
		for target, env, pos, block in itertools.product(TARGETS, ENVIRONMENTS, POSTURES, BLOCKS):
			trial_info = str(subjectNum) + "_" + make_trial_info([target, env, pos, block])
			imu_filename = "imu_" + trial_info +".csv"
			pupil_filename = "pupil_" + trial_info + ".csv"
			try:
				imu = get_refined_files(imu_filename)
				pupil = get_refined_files((pupil_filename))

			except ValueError as err:
				print(err.args)


def analyse_raw_data():
	for subjectNum in range(1, 17):
		# output = np.array([1.0, 1.0, 1.0, 1.0,1.0,1.0])
		# output = output.astype('float64')

		fileLists = get_one_subject(subjectNum)
		for target, env, pos, block in itertools.product(TARGETS, ENVIRONMENTS, POSTURES, BLOCKS):

			# just for test
			# for target, env, pos, block in itertools.product(range(0,1), ['U'], ['W'], range(4,5)):

			# 	if target ==0 and env =='W' and pos =='S' and block ==3:    should handle empty/damaged file
			# 		continue
			try:
				trial_info = [target, env, pos, block]
				[ProcessingData, HololensData, filename] = get_each_file(fileLists[0], fileLists[1], subjectNum,
				                                                         trial_info)
				filter_files(ProcessingData, HololensData, filename, subjectNum)
			# data = lookup_file(ProcessingData, HololensData, filename)
			except ValueError as err:
				print(err.args)
		print('-' * 80)


if __name__ == "__main__":
	analyse_refined_data()
	pass;
