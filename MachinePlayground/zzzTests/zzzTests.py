from numpy import *
from numpy.random import rand


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
TEST___PROGRAM___ffnn_check_gradients(100)
TEST___PROGRAM___ffnn_unskewed_classification_check_gradients(100)


###
from numpy import *
from numpy.random import rand
from MachinePlayground.Classes import Project
from MachinePlayground.Programs.PROGRAMS___ffnn import PROGRAM___ffnn, PROGRAM___ffnn_unskewed_classification
prog = PROGRAM___ffnn_unskewed_classification([4, 3], ['logistic'])
prog1 = prog.install(
    {'activations': 'activations1',
     'cost': 'cost1',
     'inputs' : 'inputs1',
     'predicted_outputs': 'predicted_outputs1',
     'signals': 'signals1',
     'target_outputs': 'target_outputs1',
     'weights': 'weights1',
     'weights_vector': 'weights_vector1',
     'positive_class_skewnesses': 'pos_skew'})
proj1 = Project()
proj1.vars =\
            {'weights_vector1': rand(5, 3),
             'weights1': {},
             'inputs1': rand(10, 4),
             'signals1': {},
             'activations1': {},
             'predicted_outputs1': array([]),
             'target_outputs1': rand(10, 3),
             'pos_skew': ones([10, 3]),
             'cost1': array([])}
proj1.programs['ffnn'] = prog1
proj1.run(('ffnn', 'forward_pass'))
proj1.run(('ffnn', 'cost'))
proj1.run(('ffnn', 'd_cost_over_d_signal_to_top_layer'))
proj1.vars
proj1.programs['ffnn'].processes
bpiece = proj1.programs['ffnn'].pieces['d_cost_over_d_signal_to_top_layer']
proj1.run(('ffnn', 'backward_pass'))


# TEST factor_product
# (examples from Coursera: "Probabilistic Graphical Models" (Daphne Koller)
from MachinePlayground.UserDefinedClasses.CLASSES___probability import Factor
from MachinePlayground.Functions.FUNCTIONS___probability import factor_product
f1 = Factor({(('a', 1), ('b', 1)): 0.5,
            (('a', 1), ('b', 2)): 0.8,
            (('a', 2), ('b', 1)): 0.1,
            (('a', 3), ('b', 1)): 0.3,
            (('a', 3), ('b', 2)): 0.9})

f2 = Factor({(('b', 1), ('c', 1)): 0.5,
            (('b', 1), ('c', 2)): 0.7,
            (('b', 2), ('c', 1)): 0.1,
            (('b', 2), ('c', 2)): 0.2})

f = factor_product(f1, f2)

