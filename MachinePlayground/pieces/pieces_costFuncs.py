from numpy import exp, log, sqrt
from MachinePlayground.Classes import Piece



def piece_squareError_half_averageOverCases():

    forwards = {'cost_squareError_half_averageOverCases':
                    [lambda a0, a1: ((a1 - a0) ** 2).sum() / (2 * a0.shape[0]),
                     {'a0': 'targetOutputs___array', 'a1': 'hypoOutputs___array'}],
                'rootMeanSquareError_averageOverCases':
                    [lambda a0, a1: sqrt(((a1 - a0) ** 2).sum() / a0.shape[0]),
                     {'a0': 'targetOutputs___array', 'a1': 'hypoOutputs___array'}]}

    backwards = {('dOverD', 'hypoOutputs___array'):
                     [lambda a0, a1: (a1 - a0) / a0.shape[0],
                      {'a0': 'targetOutputs___array', 'a1': 'hypoOutputs___array'}]}

    return Piece(forwards, backwards)



def piece_crossEntropy_binaryClasses_averageOverCases():

    forwards = {'cost_crossEntropy_binaryClasses_averageOverCases':
                    [lambda fromArr, ofArr, posSkew = [1], tiny = exp(-36):
                        - ((fromArr * log(ofArr + tiny)) / posSkew
                        + ((1. - fromArr) * log(1. - ofArr + tiny)) / (2. - posSkew)).sum() / fromArr.shape[0],
                    {'fromArr': 'targetOutputs___array', 'ofArr': 'hypoOutputs___array',
                     'posSkew': 'positiveClassSkewness'}],
                'percentConfidence':
                    [lambda: None]}

    backwards = {('dOverD', 'hypoOutputs___array'):
                    [lambda fromArr, ofArr, posSkew = [1]:
                        - ((fromArr / ofArr) / posSkew
                        - ((1. - fromArr) / (1. - ofArr)) / (2. - posSkew)) / fromArr.shape[0],
                     {'fromArr': 'targetOutputs___array', 'ofArr': 'hypoOutputs___array',
                      'posSkew': 'positiveClassSkewness'}]}

    return Piece(forwards, backwards)


