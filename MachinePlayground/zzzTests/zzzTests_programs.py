from numpy import allclose
from copy import deepcopy
from MachinePlayground._common import *
from MachinePlayground.Programs.PROGRAM___ffnn import *



def TEST___PROGRAM___ffnn_check_gradients(num_runs = 1000, rtol = 1.e-6, atol = 1.e-6):

    def cost(model, weights_vector):
        p0 = deepcopy(model)
        p0.vars['weights_vector'] = weights_vector
        p0.run('forward_pass')
        p0.run('cost')
        return p0.vars['cost']

    max_num_cases = 5
    max_num_layers = 5
    max_num_nodes_per_layer = 5

    print('\nTEST: PROGRAM___ffnn (check gradients): rtol = %g, atol = %g' %(rtol, atol))
    num_successes = 0
    for r in range(num_runs):
        L = randint(1, max_num_layers) + 1
        nums_nodes = []
        activation_functions = []
        add_biases = []
        for l in range(L):
            nums_nodes += [randint(max_num_nodes_per_layer) + 1]
            if l < L - 1:
                if l < L - 2:
                    functions = ('linear', 'logistic', 'tanh', 'softmax')
                else:
                    functions = ('linear', 'logistic', 'softmax')
                activation_functions += [functions[randint(len(functions))]]
                add_biases += [randint(2)]
        p = PROGRAM___ffnn(nums_nodes, activation_functions, add_biases)
        m = randint(max_num_cases) + 1
        p.vars['inputs'] = rand(m, nums_nodes[0])
        if activation_functions[-1] == 'softmax':
            y = rand(m, nums_nodes[-1])
            yMax = y.max(1, keepdims = True)
            p.vars['target_outputs'] = 1. * (y == yMax)
        else:
            p.vars['target_outputs'] = rand(m, nums_nodes[-1])
        weights_vector = rand(*p.vars['weights_vector'].shape)
        p.vars['weightsVector'] = weights_vector
        p.run('cost_and_d_cost_over_d_weights')
        analytic_gradients = p.vars[('DOVERD', 'cost', 'weights_vector')]
        approx_gradients = approxGradients(lambda w: cost(p, w), weights_vector)
        check = allclose(approx_gradients, analytic_gradients, rtol = rtol, atol = atol)
        if not check:
            diff = abs(approx_gradients - analytic_gradients)
            print('\nAbs Diff:')
            print(diff)
            print('Rel Diff:')
            print(diff / abs(analytic_gradients))
            print('\n')
        num_successes += check
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))