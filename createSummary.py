import sys
import os
import numpy as np
import glob
#import math


'''
Reads all the files in the directory and extracts a summary of the results. 
Expects 4 runs of each scenario
Writes the summary to a file Summary.
The highlights are the Class accuracies for Test and best_test

Typical input at the end of the results files looks like: 

==> frost/cifar10.3@10-1Iter3175U7CL6C0.95WD8e-4WU2Wclr1BS32LR0.06M0.9T1Ad.d.dB4D0.2BF16_3 <==
Epochs 64, kimg 3136   accuracy train/valid/test/best_test  100.00  100.00  90.53  90.53
Total training time 7.4010 hours

'''

#numFiles = int(sys.argv[2])
numFiles = 4

bestAcc = [0]*numFiles

print("=> Writing out files ....")
filename = 'ResultsSummary'
print(filename)
fileOut = open(filename,'w')

files = os.listdir('.')
listing = glob.glob('./*0')

listing.sort()
#print(listing)
for j in range(len(listing)):
#    classAcc = ['', '', '', '']
#    midway   = np.zeros(4, dtype=float)
    bestAcc  = np.zeros(4, dtype=float)
    classAcc  = np.zeros(10, dtype=float)

#    print(listing[j])
    for i in range(0,numFiles):
        name = listing[j]
        name = name[:-1] + str(i)
        seen = 0
        testclass = ""
#        print(name," exits ",os.path.isfile(name))
        if os.path.isfile(name):
            with open(name,"r") as f:
                for line in f:
                    if (line.find("est set class") > 0):
                        testclass = line
                        seen = 1
                    if (line.find("accuracy train/valid/test/best_test") > 0):
                        pref, post = line.split("best_test")
                        accs = post.split("  ")
                        bestAcc[i] = float(accs[4][:-1])
        if seen == 1:
            accs = testclass.split(",")
            pref,post = accs[0].split("[")
            accs[0] = post
            pref,post = accs[9].split("]")
            accs[9] = pref
            for k in range(10):
                classAcc[k] += float(accs[k])

    classAcc /= numFiles
    print(classAcc)

#    midAcc = np.mean(midway)
#    print("seen ", seen)
    midway = np.sort(bestAcc)
    midAcc = "{:.2f}".format(np.mean(midway[1:]))
#    acc = np.mean(bestAcc)
    acc = "{:.2f}".format(np.mean(bestAcc))
    accSTD = "{:.2f}".format(np.std(bestAcc))
    print(name," ",midAcc," ",bestAcc," mean, std= ",acc,accSTD)

#    for i in range(0,numFiles):
#        print(classAcc[i])
#        print(numTrainPerClass[i])

#    fileOut.write('{:f}'.format(bestAcc)
    fileOut.write(name+"  "+midAcc+"  [")
    for i in range(0,numFiles):
        fileOut.write(str(bestAcc[i])+" ")
    fileOut.write("] Mean= "+acc)
    fileOut.write(" STD= "+accSTD)
    fileOut.write("\n")
#    for i in range(0,numFiles):
#        fileOut.write(classAcc[i])
#        accs = classAcc[i].split(",")
#        try:
#            for x in accs:
#                fileOut.write('{0:0.2f},  '.format(float(x)))
#        except:
#            pass
#        fileOut.write("\n")
#    for i in range(0,numFiles):
#        fileOut.write(numTrainPerClass[i])
#        fileOut.write("\n")
fileOut.close()
exit(1)
