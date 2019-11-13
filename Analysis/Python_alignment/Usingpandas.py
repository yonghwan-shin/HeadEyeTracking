from subjectHandling import *
from oneEuroFilter import *
import itertools

# Full test
TARGETS = range(8)
ENVIRONMENTS = ['U', 'W']
POSTURES = ['W', 'S']
BLOCKS = range(5)


# for subjectNum in range(2, 3):
for subjectNum in range(1,17):
	# output = np.array([1.0, 1.0, 1.0, 1.0,1.0,1.0])
	# output = output.astype('float64')

	fileLists = get_one_subject(subjectNum)
	for target,env,pos,block in itertools.product(TARGETS,ENVIRONMENTS,POSTURES,BLOCKS):

	# just for test
	# for target, env, pos, block in itertools.product(range(0,1), ['U'], ['W'], range(4,5)):

	# 	if target ==0 and env =='W' and pos =='S' and block ==3:    should handle empty/damaged file
	# 		continue
		try:
			trial_info = [target, env, pos, block]
			[ProcessingData, HololensData, filename] = get_each_file(fileLists[0], fileLists[1], subjectNum, trial_info)
			filter_files(ProcessingData,HololensData,filename,subjectNum)
			# data = lookup_file(ProcessingData, HololensData, filename)
		except ValueError as err:
			print(err.args)



	# np.savetxt('minmax'+str(subjectNum)+'.csv', output, delimiter=',', fmt='%1.4f')
	print('-' * 80)


