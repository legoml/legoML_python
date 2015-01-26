from numpy import array, atleast_2d, exp, log, sqrt
from MachinePlayground.Classes import Piece



def piece_squareError_half_averageOverCases():

    forwards = {'cost_squareError_half_averageOverCases':
                    [lambda a0, a1: ((a1 - a0) ** 2).sum() / (2 * a0.shape[0]),
                     {'a0': 'targetOutputs___array',
                      'a1': 'hypoOutputs___array'}],
                'rootMeanSquareError_averageOverCases':
                    [lambda a0, a1: sqrt(((a1 - a0) ** 2).sum() / a0.shape[0]),
                     {'a0': 'targetOutputs___array',
                      'a1': 'hypoOutputs___array'}]}

    backwards = {('dOverD', 'cost_squareError_half_averageOverCases', 'hypoOutputs___array'):
                     [lambda a0, a1: (a1 - a0) / a0.shape[0],
                      {'a0': 'targetOutputs___array',
                       'a1': 'hypoOutputs___array'}]}

    return Piece(forwards, backwards)



def piece_crossEntropy_binaryClasses_averageOverCases():

    forwards = {'cost_crossEntropy_binaryClasses_averageOverCases':
                    [lambda fromArr, ofArr, posSkew = array([1]), tiny = exp(-36):
                        - ((fromArr * log(ofArr + tiny)) / posSkew
                        + ((1. - fromArr) * log(1. - ofArr + tiny)) / (2. - posSkew)).sum() / fromArr.shape[0],
                    {'fromArr': 'targetOutputs___array',
                     'ofArr': 'hypoOutputs___array',
                     'posSkew': 'positiveClassSkewnesses'}]}

    backwards = {('dOverD', 'cost_crossEntropy_binaryClasses_averageOverCases', 'hypoOutputs___array'):
                    [lambda fromArr, ofArr, posSkew = array([1]):
                        - ((fromArr / ofArr) / posSkew
                        - ((1. - fromArr) / (1. - ofArr)) / (2. - posSkew)) / fromArr.shape[0],
                     {'fromArr': 'targetOutputs___array',
                      'ofArr': 'hypoOutputs___array',
                      'posSkew': 'positiveClassSkewnesses'}]}

    return Piece(forwards, backwards)



def piece_crossEntropy_multiClasses_averageOverCases():

    forwards = {'cost_crossEntropy_multiClasses_averageOverCases':
                    [lambda fromMat, ofMat, skew = atleast_2d(array([1])), tiny = exp(-36):
                        - ((fromMat * log(ofMat + tiny))
                           / ((fromMat.shape[1] / skew.sum(1, keepdims = True)) * skew)).sum() / fromMat.shape[0],
                     {'fromMat': 'targetOutputs___matrixRowsForCases',
                      'ofMat': 'hypoOutputs___matrixRowsForCases',
                      'skew': 'classSkewnesses'}]}

    backwards = {('dOverD', 'cost_crossEntropy_multiClasses_averageOverCases', 'hypoOutputs___matrixRowsForCases'):
                     [lambda fromMat, ofMat, skew = atleast_2d(array([1])):
                        - ((fromMat / ofMat)
                           / ((fromMat.shape[1] / skew.sum(1, keepdims = True)) * skew)) / fromMat.shape[0],
                      {'fromMat': 'targetOutputs___matrixRowsForCases',
                      'ofMat': 'hypoOutputs___matrixRowsForCases',
                      'skew': 'classSkewnesses'}]}

    return Piece(forwards, backwards)