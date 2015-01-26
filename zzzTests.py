from importlib import *
from math import *
from numpy import *
from MachinePlayground._common import *
from MachinePlayground.Classes import *
from MachinePlayground.funcs import *


f1 = [{'a1': 'i1', 'a2': 'i2'}, lambda a1, a2: a1 + a2, 's12']
f2 = [{'b1': 'i2', 'b2': 'i3'}, lambda b1, b2: b1 * b2, 'p23']
p = Piece(f1, f2)
d = {'i1': 3, 'i2': 4, 'i3': 5, 'i4': 6}
p.inKeys
p.func(d)
p1 = p.installPiece({'in1': 'i1', 'in2': 'i2', 'in3': 'i3'}, {'sum12': 's12', 'prod23': 'p23'})
p.inKeys
p.outKeys
p1.inKeys
d1 = {'in1': 3, 'in2': 4, 'in3': 5, 'in4': 6}
p1.func(d1)

d_toUpdate = {'a': [1, 2, 'c'], 'b': {'d': 10, 'e': 20}}
d_withValues = {'u': {'x': 100, 'y': 2000}, 'v': [123, 456, 'mnp']}
mapDict = {('a', 1): ('u', 'y'), ('b', 'e'): 'v'}
updateDictValues(d_toUpdate, mapDict, d_withValues)



p1 = installPiece

biases
inputs
hypoOutputs
targetOutputs
weights
forwardFuncs
backwardFuncs
costFuncs
activations
signals



# test BIAS TERMS
from numpy import *
from MachinePlayground.funcs import biasTerms as bias

mat = array([[10, 20, 30], [4, 5, 6], [-7, -8, -9], [10, 11, 12]])
mat.max(0)

print(bias.addBiasElements(mat))
print(bias.addBiasElements(mat, [0, 0]))
print(bias.addBiasElements(mat, [0, 2]))
print(bias.addBiasElements(mat, [3, 0]))

print(bias.deleteBiasElements(mat))
print(bias.deleteBiasElements(mat, [0]))
print(bias.deleteBiasElements(mat, [3]))
print(bias.deleteBiasElements(mat, [0, 2]))

print(bias.zeroBiasElements(mat))
print(bias.zeroBiasElements(mat, [0]))
print(bias.zeroBiasElements(mat, [3]))
print(bias.zeroBiasElements(mat, [0, 2]))



# test ACTIVATION FUNCS
from numpy import *
from MachinePlayground.funcs import activationFuncs as activ

m = array([[0.1, 0.3, 1], [1.3, 2, 3]])

print(activ.activations(m))
print(activ.dActivations_over_dSignals(m))

print(activ.activations(m, nameOfActivationFunc = 'logistic'))
print(activ.dActivations_over_dSignals(m, nameOfActivationFunc = 'logistic'))

print(activ.activations(m, nameOfActivationFunc = 'tanh'))
print(activ.dActivations_over_dSignals(m, nameOfActivationFunc = 'tanh'))

print(activ.activations(m, nameOfActivationFunc = 'softmax'))
print(activ.dActivations_over_dSignals(m, nameOfActivationFunc = 'softmax'))

# test PIECE
from MachinePlayground.pieces.zzzTESTS import *
TEST_piece_equal()
TEST_piece_multiplyMatrices_ofInputsAndWeights()
TEST_piece_linear()
TEST_piece_logistic()
TEST_piece_tanh()
TEST_piece_softmax()
TEST_piece_squareError_half_averageOverCases()
TEST_piece_crossEntropy_binaryClasses_averageOverCases()
TEST_piece_crossEntropy_multiClasses_averageOverCases()

# test FFNN
from MachinePlayground.programs.zzzTESTS import *
TEST_ffNN_checkGradients(1000, 1.e-3, 1.e-6)