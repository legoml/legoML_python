from numpy.random import *
from MBALearnsToCode.Pieces.PIECES___zzz_common_functions import *
from MBALearnsToCode.Pieces.PIECES___cost_functions import *
from MBALearnsToCode.Pieces.PIECES___regularization_functions import *



def TEST___PIECE___equal(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___equal
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        inp = rand(*dim_sizes)
        num_successes += p.check_gradients({'inputs': inp})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___matrix_product_of_inputs_and_weights(num_runs = 1000):
    max_num_cases = 9
    max_input_dim_size = 9
    max_output_dim_size = 9
    f = PIECE___matrix_product_of_inputs_and_weights
    print('\nTEST: ', f.__doc__, sep = '')
    num_successes = 0
    for r in range(num_runs):
        add_bias = randint(2)
        p = f(add_bias)
        m = randint(max_num_cases) + 1
        nI = randint(max_input_dim_size) + 1
        nO = randint(max_output_dim_size) + 1
        inp = rand(m, nI)
        w = rand(nI + add_bias, nO)
        num_successes += p.check_gradients({'inputs': inp, 'weights': w})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___linear(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___linear
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        inp = rand(*dim_sizes)
        num_successes += p.check_gradients({'inputs': inp})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___logistic(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___logistic
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        inp = rand(*dim_sizes)
        num_successes += p.check_gradients({'inputs': inp})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___logistic_with_temperature(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___logistic_with_temperature
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        inp = rand(*dim_sizes)
        temp = 3 * rand()
        num_successes += p.check_gradients({'inputs': inp, 'temperature': temp})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___tanh(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___tanh
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        inp = rand(*dim_sizes)
        num_successes += p.check_gradients({'inputs': inp})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___softmax(num_runs = 1000):
    max_num_cases = 9
    max_dim_size = 9
    f = PIECE___softmax
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        m = randint(max_num_cases) + 1
        n = randint(max_dim_size) + 1
        inp = rand(m, n)
        num_successes += p.check_gradients({'inputs': inp})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___softmax_with_temperature(num_runs = 1000):
    max_num_cases = 9
    max_dim_size = 9
    f = PIECE___softmax_with_temperature
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        m = randint(max_num_cases) + 1
        n = randint(max_dim_size) + 1
        inp = rand(m, n)
        temp = 3 * rand()
        num_successes += p.check_gradients({'inputs': inp,
                                            'temperature': temp})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___average_half_square_error(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___average_half_square_error
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        a0 = rand(*dim_sizes)
        a1 = rand(*dim_sizes)
        num_successes += p.check_gradients({'target_outputs': a0, 'predicted_outputs': a1})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___root_mean_square_error(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___root_mean_square_error
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        a0 = rand(*dim_sizes)
        a1 = rand(*dim_sizes)
        num_successes += p.check_gradients({'target_outputs': a0, 'predicted_outputs': a1})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___root_mean_square_error_from_average_half_square_error(num_runs = 1000):
    f = PIECE___root_mean_square_error_from_average_half_square_error
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        avg_half_se = array(rand())
        num_successes += p.check_gradients({'average_half_square_error': avg_half_se})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___average_binary_class_cross_entropy(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___average_binary_class_cross_entropy
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        from_arr = rand(*dim_sizes)
        of_arr = rand(*dim_sizes)
        num_successes += p.check_gradients({'target_outputs': from_arr,
                                            'predicted_outputs': of_arr})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___average_unskewed_binary_class_cross_entropy(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___average_unskewed_binary_class_cross_entropy
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        from_arr = rand(*dim_sizes)
        of_arr = rand(*dim_sizes)
        pos_skew = 2 * rand(*dim_sizes)
        num_successes += p.check_gradients({'target_outputs': from_arr, 'predicted_outputs': of_arr,
                                          'positive_class_skewnesses': pos_skew})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___average_multi_class_cross_entropy(num_runs = 1000):
    max_num_cases = 9
    max_dim_size = 9
    f = PIECE___average_multi_class_cross_entropy
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        m = randint(max_num_cases) + 1
        n = randint(max_dim_size) + 1
        from_arr = rand(m, n)
        of_arr = rand(m, n)
        num_successes += p.check_gradients({'target_outputs': from_arr,
                                            'predicted_outputs': of_arr})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___average_unskewed_multi_class_cross_entropy(num_runs = 1000):
    max_num_cases = 9
    max_dim_size = 9
    f = PIECE___average_unskewed_multi_class_cross_entropy
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        m = randint(max_num_cases) + 1
        n = randint(max_dim_size) + 1
        from_arr = rand(m, n)
        of_arr = rand(m, n)
        skew = rand(1, n)
        num_successes += p.check_gradients({'target_outputs': from_arr,
                                            'predicted_outputs': of_arr,
                                            'multi_class_skewnesses': skew})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___l1_weight_regularization(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___l1_weight_regularization
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        w = rand(*dim_sizes)
        num_successes += p.check_gradients({'weights': w})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PIECE___l2_weight_regularization(num_runs = 1000):
    max_num_dims = 3
    max_dim_size = 9
    f = PIECE___l2_weight_regularization
    print('\nTEST: ', f.__doc__, sep = '')
    p = f()
    num_successes = 0
    for r in range(num_runs):
        num_dims = randint(max_num_dims) + 1
        dim_sizes = []
        for d in range(num_dims):
            dim_sizes += [randint(max_dim_size) + 1]
        w = rand(*dim_sizes)
        num_successes += p.check_gradients({'weights': w})
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))