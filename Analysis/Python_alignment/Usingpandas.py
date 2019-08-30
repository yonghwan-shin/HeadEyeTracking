# from fileHandling import *
from subjectHandling import *

for subjectNum in range(1,2):
    # for subjectNum in range(1, 2):
    fileLists = getOneSubject(subjectNum)

    # for fileCount in range(0,len(fileLists[0])):
    for fileCount in range(0,3):
        Files = getOneFile(fileLists[0],fileLists[1],subjectNum,fileCount)
        analyzeFile(Files[0],Files[1])

    print('-'*80)