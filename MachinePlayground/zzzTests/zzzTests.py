from numpy import *

# test PIECE
from MachinePlayground.zzzTests.zzzTests_pieces import *
TEST___PIECE___equal()
TEST___PIECE___matrix_product_of_inputs_and_weights()
TEST___PIECE___linear()
TEST___PIECE___logistic()
TEST___PIECE___logistic_with_temperature()
TEST___PIECE___tanh()
TEST___PIECE___softmax()
TEST___PIECE___softmax_with_temperature()
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
TEST___PROGRAM___ffnn_check_gradients(3)
TEST___PROGRAM___ffnn_unskewed_classification_check_gradients(300)

#
from MachinePlayground.Programs.PROGRAMS___ffnn import PROGRAM___ffnn
prog = PROGRAM___ffnn([4, 3], ['logistic'])
prog1 = prog.install(
    {'activations': 'activations1',
     'cost': 'cost1',
     'inputs' : 'inputs1',
     'predicted_outputs': 'predicted_outputs1',
     'signals': 'signals1',
     'target_outputs': 'target_outputs1',
     'weights': 'weights1',
     'weights_vector': 'weights_vector1'})
prog1.processes['forward_pass'].steps[0]