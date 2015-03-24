from numpy import *
from importlib import reload

# test PIECE
from MachinePlayground.zzzTests.zzzTests_pieces import *
TEST___PIECE___equal()
TEST___PIECE___matrix_product_of_inputs_and_weights()
TEST___PIECE___linear()
TEST___PIECE___logistic()
TEST___PIECE___tanh()
TEST___PIECE___softmax()
TEST___PIECE___average_half_square_error()
TEST___PIECE___root_mean_square_error()
TEST___PIECE___root_mean_square_error_from_average_half_square_error()
TEST___PIECE___average_binary_class_cross_entropy()
TEST___PIECE___average_unskewed_binary_class_cross_entropy()
TEST___PIECE___average_multi_class_cross_entropy()
TEST___PIECE___average_unskewed_multi_class_cross_entropy()
TEST___PIECE___l1_weight_regularization()
TEST___PIECE___l2_weight_regularization()

# test FFNN
from MachinePlayground.zzzTests.zzzTests_programs import *
TEST___PROGRAM___ffnn_check_gradients(30)

# test Project
import MachinePlayground.Classes as C

reload(C)
p = C.Project()