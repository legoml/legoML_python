from numpy import allclose
from numpy.random import *
from MachinePlayground.Classes import *
from MachinePlayground.funcs.zzzMiscFuncs import *


#def group_ffNN(*args, **kwargs):



#    return Group()


#def process_forwardPass(numLayers = 1):
#    i = 0
#    p[0] = Piece([{'X': 'inputs'}, lambda X: identity(X), ('activations', 0)])
#    while i <= numLayers:
#        i += 1
#        f0 = [{'activations', i}, ]
#        f1 =
#        p[i] = Piece(f0, f1, f2)



#    return


#def process_computeCost:
#    return

#def process_backwardPass:
#    return



def ffNN_gradientCheck():

    maxNumNodesPerLayer = 6
    maxNumLayers = 6
    maxNumCases = 9

    m = randint(maxNumCases) + 1
    nI = randint(maxNumNodesPerLayer) + 1
    nL = randint(maxNumLayers) + 1

    ffNN = {}
    ffNN['inputs'] = rand(m, nI)


    for l in range(nL):
        ffNN['weights'][l] = []

    ffNN['weights']

