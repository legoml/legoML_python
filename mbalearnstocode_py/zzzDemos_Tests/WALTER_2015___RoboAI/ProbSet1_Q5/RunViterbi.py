import sys
from DataSet import *
from mbalearnstocode_py.UserDefinedClasses.CLASSES___HiddenMarkovModels import HMM, Viterbi


# This defines a simple class for running your Viterbi code. As with
# all the code, feel free to modify it as you see fit, or to write
# your own outright.


class RunViterbi(object):

    def __init__(self):
        self.maxSequence = []


    def trainHMM(self, filename):
        print "Reading training data from %s" % (filename)

        # Read in the training data from the file
        dataset = DataSet(filename)
        dataset.readFile()

        # Instatiate and train the HMM
        self.hmm = HMM(dataset.numStates, dataset.numOutputs, dataset.trainState, dataset.trainOutput)
        self.hmm.train()
        
        return

    def estMaxSequence(self, filename):

        print "Reading testing data from %s" % (filename)

        # Read in the testing dta from the file
        dataset = DataSet(filename)
        dataset.readFile()

        # Run Viterbi to estimate most likely sequence
        viterbi = Viterbi(self.hmm)
        self.maxSequence = viterbi.mostLikelySequence(dataset.testOutput)


if __name__ == '__main__':

    # This function should be called with two arguments: trainingdata.txt testingdata.txt
    viterbi = RunViterbi()
    viterbi.trainHMM(sys.argv[1])
    


    




