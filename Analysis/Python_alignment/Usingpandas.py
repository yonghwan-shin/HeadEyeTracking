from subjectHandling import *

for subjectNum in range(3, 4):
    fileLists = get_one_subject(subjectNum)

    # for fileCount in range(0,len(fileLists[0])):
    for i in range(2,3):
        trial_info = ['0', 'U', 'W', i]
        [ProcessingData, HololensData, filename] = get_each_file(fileLists[0], fileLists[1], subjectNum, trial_info)
        lookup_file(ProcessingData, HololensData, filename)

    print('-' * 80)
