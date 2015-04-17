import csv
import numpy as np


class DataSet(object):
    def __init__(self, filename):
        # The following are some variables that may be necessary or
        # useful. You may find that you need/want to add other variables.
        self.filename = filename
        self.numStates = 16
        self.numOutputs = 4

        # The set of all training state sequences where trainState[i]
        # is an array of state sequences for the ith training
        # sequence. The corresponding output sequence is at trainOutput[i]
        self.trainState = []

        # The set of all training observation sequences where trainOutput[i]
        # is an array of output sequences for the ith training
        # sequence. The corresponding state sequence is at trainState[i]
        self.trainOutput = []

         # The set of all testing state sequences where testState[i]
        # is an array of state sequences for the ith test
        # sequence. The corresponding output sequence is at testOutput[i]
        self.testState = []

        # The set of all testing observation sequences where testOutput[i]
        # is an array of output sequences for the ith test
        # sequence. The corresponding state sequence is at testState[i]
        self.testOutput = []


        # Assume a 4x4 world. Map (x,y) pairs to integers
        # with (1,1) being 0, (1,2) being 1, ...
        self.xyToInt = np.zeros((4,4))

        # Map 'r','g','b','y' color to integer
        self.obsToInt = {'r': 0, 'g': 1, 'b': 2, 'y': 3}

        idx = 0
        for i in range(4):
            for j in range(4):
                self.xyToInt[i,j] = idx
                idx+=1

        


    # This function reads in the file and populates the training state
    # and output sequences
    def readFile(self):

        allStateSeq = []
        allObsSeq = []


        # Your code goes here
        print "Please add code"
        
                
