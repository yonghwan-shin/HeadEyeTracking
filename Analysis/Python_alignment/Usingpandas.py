# from fileHandling import *
from subjectHandling import *

for subjectNum in range(3, 4):
    fileLists = getOneSubject(subjectNum)

    # for fileCount in range(0,len(fileLists[0])):
    # for fileCount in range(1, 20):
    for i in range(5):
        Files = getOneFile(fileLists[0], fileLists[1], subjectNum, ['0','W','W',i])
        analyzeFile(Files[0], Files[1],Files[2])

    print('-' * 80)
