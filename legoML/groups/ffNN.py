from legoML.funcs.zzzMiscFuncs import *
from legoML.Classes import *

def group_ffNN(*args, **kwargs):



    return Group()


def process_forwardPass(numLayers = 1):
    i = 0
    p[0] = Piece([{'X': 'inputs'}, lambda X: identity(X), ('activations', 0)])
    while i <= numLayers:
        i += 1
        f0 = [{'activations', i}, ]
        f1 =
        p[i] = Piece(f0, f1, f2)



    return


def process_computeCost:
    return

def process_backwardPass:
    return