from numpy import *
m = array([[10, 20, 30], [4, 5, 6], [-7, -8, -9], [10, 11, 12]])

# test PIECE
from MachinePlayground.zzzTests.zzzTests_pieces import *
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
from MachinePlayground.zzzTests.zzzTests_programs import *
TEST_ffNN_checkGradients(300)

# test Project
import MachinePlayground.Classes as C
from importlib import reload
reload(C)
p = C.Project()