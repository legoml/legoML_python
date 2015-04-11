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




from frozen_dict import FrozenDict as fdict
from MachinePlayground.UserDefinedClasses.CLASSES___probability import Factor

# TEST factor_product
# (examples from Coursera: "Probabilistic Graphical Models" (Daphne Koller)
# expected result:
# (('a', 1), ('b', 1), ('c', 1)): 0.25
# (('a', 1), ('b', 1), ('c', 2)): 0.35
# (('a', 1), ('b', 2), ('c', 1)): 0.08
# (('a', 1), ('b', 2), ('c', 2)): 0.16
# (('a', 2), ('b', 1), ('c', 1)): 0.05
# (('a', 2), ('b', 1), ('c', 2)): 0.07
# (('a', 2), ('b', 2), ('c', 1)): 0
# (('a', 2), ('b', 2), ('c', 2)): 0
# (('a', 3), ('b', 1), ('c', 1)): 0.15
# (('a', 3), ('b', 1), ('c', 2)): 0.21
# (('a', 3), ('b', 1), ('c', 2)): 0.21
# (('a', 3), ('b', 2), ('c', 1)): 0.09
# (('a', 3), ('b', 2), ('c', 2)): 0.18
f1 = Factor({fdict(a=1, b=1): 0.5,
             fdict(a=1, b=2): 0.8,
             fdict(a=2, b=1): 0.1,
             fdict(a=3, b=1): 0.3,
             fdict(a=3, b=2): 0.9})

f2 = Factor({fdict(b=1, c=1): 0.5,
             fdict(b=1, c=2): 0.7,
             fdict(b=2, c=1): 0.1,
             fdict(b=2, c=2): 0.2})

f = f1.product(f2)
g = f.margin(('b',))


# ROBOTICS and AI HOMEWORK 01, QUESTION 01
f_C = Factor({fdict(C='a'): 0.3,
              fdict(C='n'): 0.5,
              fdict(C='l'): 0.2})
f_T1_on_C = Factor({fdict(C='a', T1=0): 0.3,
                    fdict(C='a', T1=1): 0.7,
                    fdict(C='n', T1=0): 0.5,
                    fdict(C='n', T1=1): 0.5,
                    fdict(C='l', T1=0): 0.8,
                    fdict(C='l', T1=1): 0.2})
f_T2_on_C = Factor({fdict(C='a', T2=0): 0.2,
                    fdict(C='a', T2=1): 0.8,
                    fdict(C='n', T2=0): 0.6,
                    fdict(C='n', T2=1): 0.4,
                    fdict(C='l', T2=0): 0.9,
                    fdict(C='l', T2=1): 0.1})
f_D_on_T1_T2 = Factor({fdict(T1=0, T2=0, D=0): 1.0,
                       fdict(T1=0, T2=0, D=1): 0.0,
                       fdict(T1=0, T2=1, D=0): 1.0,
                       fdict(T1=0, T2=1, D=1): 0.0,
                       fdict(T1=1, T2=0, D=0): 0.0,
                       fdict(T1=1, T2=0, D=1): 1.0,
                       fdict(T1=1, T2=1, D=0): 0.0,
                       fdict(T1=1, T2=1, D=1): 1.0})
f_T1 = (f_C.product(f_T1_on_C)).margin(['C'])
f_T2 = (f_C.product(f_T2_on_C)).margin(['C'])
f_T1_T2 = (f_C.product(f_T1_on_C, f_T2_on_C)).margin(['C'])
f_T1_f_T2 = f_T1.product(f_T2)
f_T1_T2_on_C_equal_n = ((f_C.product(f_T1_on_C, f_T2_on_C)).condition(dict(C='n'))).normalize()

f_T1_on_C_equal_n = f_T1_on_C.condition(dict(C='n'))
f_T2_on_C_equal_n = f_T2_on_C.condition(dict(C='n'))
f_T1_T2_on_D_equal_1 = f_D_on_T1_T2.condition(dict(D=1))
f_T1_T2_on_C_equal_n_and_D_equal_1 = (f_T1_on_C_equal_n.product(f_T2_on_C_equal_n,
                                                                f_T1_T2_on_D_equal_1)).normalize()
f_T1_T2_on_C_equal_n_and_D_equal_1___alternative =\
    (f_C.product(f_T1_on_C, f_T2_on_C, f_D_on_T1_T2).condition(dict(C='n', D=1))).normalize()


# ROBOTICS and AI HOMEWORK 01, QUESTION 02
alpha = 0.6

f_S0_S1 = Factor({fdict(S0=0, S1=0): 0.9,
                  fdict(S0=0, S1=1): 0.1,
                  fdict(S0=1, S1=0): 0.1,
                  fdict(S0=1, S1=1): 0.9})
f_S1_S2 = Factor({fdict(S1=0, S2=0): alpha,
                  fdict(S1=0, S2=1): 1.0 - alpha,
                  fdict(S1=1, S2=0): 1.0,
                  fdict(S1=1, S2=1): alpha})
f_S2_S3 = Factor({fdict(S2=0, S3=0): alpha,
                  fdict(S2=0, S3=1): 1.0 - alpha,
                  fdict(S2=1, S3=0): 1.0,
                  fdict(S2=1, S3=1): alpha})
f_S3_S4 = Factor({fdict(S3=0, S4=0): 0.9,
                  fdict(S3=0, S4=1): 0.1,
                  fdict(S3=1, S4=0): 0.1,
                  fdict(S3=1, S4=1): 0.9})

f_S1_on_S0_equal_1 = f_S0_S1.condition(dict(S0=1))
f_S3_on_S4_equal_0 = f_S3_S4.condition(dict(S4=0))
f_S1_on_S0_equal_1_and_S4_equal_0 = ((f_S1_on_S0_equal_1.product(f_S1_S2, f_S2_S3,
                                                                 f_S3_on_S4_equal_0)).margin(['S2', 'S3'])).normalize()



