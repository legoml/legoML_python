import numpy as np

class HMM(object):
    # Construct an HMM with the following set of variables
    #    numStates:          The size of the state space
    #    numOutputs:         The size of the output space
    #    trainStates[i][j]:  The jth element of the ith state sequence
    #    trainOutputs[i][j]: Similarly, for output
    def __init__(self, numStates, numOutputs, states, outputs):
        self.numStates = numStates
        self.numOutputs = numOutputs
        self.states = states
        self.outputs = outputs


        # Your code goes here
        print "Please add code"


    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self):

        # Your code goes here
        print "Please add code"

    # Returns the log probability associated with a transition from
    # the dummy start state to the given state according to this HMM
    def getLogStartProb (state):

        # Your code goes here
        print "Please add code"

    # Returns the log probability associated with a transition from
    # fromState to toState according to this HMM
    def getLogTransProb (fromState, toState):

        # Your code goes here
        print "Please add code"

    # Returns the log probability of state state emitting output
    # output
    def getLogOutputProb (state, output):

        # Your code goes here
        print "Please add code"



# This is a template for a Vitirbi class that can be used to compute
# most likely sequences.

class Viterbi(object):
    # Construct an instance of the viterbi class
    # based upon an instantiatin of the HMM class
    def __init__(self, hmm):
        self.hmm = hmm

        # Your code goes here


    # Returns the most likely state sequence for a given output
    # (observation) sequence, i.e.,
    #    arg max_{X_1, X_2, ..., X_T} P(X_1,...,X_T | Z_1,...Z_T)
    # according to the HMM model that was passed to the constructor.
    def mostLikelySequence(output):

        # Your code goes here
        print "Your code goes here"

