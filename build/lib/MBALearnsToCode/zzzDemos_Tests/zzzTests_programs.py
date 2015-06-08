from copy import deepcopy

from numpy import allclose, array, ones
from numpy.random import rand, randint

from MBALearnsToCode.Functions.FUNCTIONS___zzzUtility import approx_gradients
from MBALearnsToCode.Programs import PROGRAM___ffnn, PROGRAM___ffnn_unskewed_classification
from MBALearnsToCode import Project


def TEST___PROGRAM___ffnn_check_gradients(num_runs = 1000, rtol = 1.e-6, atol = 1.e-6):

    def cost(model, w):
        p0 = deepcopy(model)
        p0.vars['w_v'] = w
        p0.play(('ffnn', 'forward_pass'))
        p0.play(('ffnn', 'cost'))
        return p0.vars['c']

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
        program = PROGRAM___ffnn(nums_nodes, activation_functions, add_biases)
        num_layers = len(nums_nodes)

        for l in range(len(add_biases) + 1, num_layers - 1):
            add_biases += add_biases[-1]
        num_weights = 0
        for l in range(num_layers - 1):
            num_weights += (nums_nodes[l] + add_biases[l]) * nums_nodes[l + 1]

        project = Project()
        project.vars =\
            {'w_v': array([]),
             'w': {},
             'inp': array([]),
             'sig': {},
             'activ': {},
             'hypo': array([]),
             'tgt': array([]),
             'c': array([])}
        project.programs['ffnn'] = program.install(
            {'weights_vector': 'w_v',
             'weights': 'w',
             'inputs': 'inp',
             'signals': 'sig',
             'activations': 'activ',
             'predicted_outputs': 'hypo',
             'target_outputs': 'tgt',
             'cost': 'c'})

        m = randint(max_num_cases) + 1
        project.vars['inp'] = rand(m, nums_nodes[0])
        if activation_functions[-1] == 'softmax':
            y = rand(m, nums_nodes[-1])
            yMax = y.max(1, keepdims = True)
            project.vars['tgt'] = 1. * (y == yMax)
        else:
            project.vars['tgt'] = rand(m, nums_nodes[-1])
        weights_vector = rand(num_weights)
        project.vars['w_v'] = weights_vector
        project.play(('ffnn', 'cost_and_d_cost_over_d_weights'))
        analytic_gradients = project.vars[('DOVERD', 'c', 'w_v')]
        numerical_gradients = approx_gradients(lambda w: cost(project, w), weights_vector)
        check = allclose(numerical_gradients, analytic_gradients, rtol = rtol, atol = atol)
        if not check:
            diff = abs(numerical_gradients - analytic_gradients)
            print('\nAbs Diff:')
            print(diff)
            print('Rel Diff:')
            print(diff / abs(analytic_gradients))
            print('\n')
        num_successes += check
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))



def TEST___PROGRAM___ffnn_unskewed_classification_check_gradients(num_runs = 1000, rtol = 1.e-6, atol = 1.e-6):

    def cost(model, w):
        p0 = deepcopy(model)
        p0.vars['w_v'] = w
        p0.play(('ffnn', 'forward_pass'))
        p0.play(('ffnn', 'cost'))
        return p0.vars['c']

    max_num_cases = 5
    max_num_layers = 5
    max_num_nodes_per_layer = 5

    print('\nTEST: PROGRAM___ffnn_unskewed_classification (check gradients): rtol = %g, atol = %g' %(rtol, atol))
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
                    functions = ('logistic', 'softmax')
                activation_functions += [functions[randint(len(functions))]]
                add_biases += [randint(2)]
        program = PROGRAM___ffnn_unskewed_classification(nums_nodes, activation_functions, add_biases)
        num_layers = len(nums_nodes)

        for l in range(len(add_biases) + 1, num_layers - 1):
            add_biases += add_biases[-1]
        num_weights = 0
        for l in range(num_layers - 1):
            num_weights += (nums_nodes[l] + add_biases[l]) * nums_nodes[l + 1]

        project = Project()
        project.vars =\
            {'w_v': array([]),
             'w': {},
             'inp': array([]),
             'sig': {},
             'activ': {},
             'hypo': array([]),
             'tgt': array([]),
             'pos_skew': array([]),
             'multi_skew': array([]),
             'c': array([])}
        project.programs['ffnn'] = program.install(
            {'weights_vector': 'w_v',
             'weights': 'w',
             'inputs': 'inp',
             'signals': 'sig',
             'activations': 'activ',
             'predicted_outputs': 'hypo',
             'target_outputs': 'tgt',
             'positive_class_skewnesses': 'pos_skew',
             'multi_class_skewnesses': 'multi_skew',
             'cost': 'c'})

        m = randint(max_num_cases) + 1
        project.vars['inp'] = rand(m, nums_nodes[0])
        if activation_functions[-1] == 'softmax':
            y = rand(m, nums_nodes[-1])
            yMax = y.max(1, keepdims = True)
            project.vars['tgt'] = 1. * (y == yMax)
            project.vars['multi_skew'] = ones([m, nums_nodes[-1]])
        else:
            project.vars['tgt'] = rand(m, nums_nodes[-1])
            project.vars['pos_skew'] = ones([m, nums_nodes[-1]]) ####
        weights_vector = rand(num_weights)
        project.vars['w_v'] = weights_vector
        project.play(('ffnn', 'cost_and_d_cost_over_d_weights'))
        analytic_gradients = project.vars[('DOVERD', 'c', 'w_v')]
        numerical_gradients = approx_gradients(lambda w: cost(project, w), weights_vector)
        check = allclose(numerical_gradients, analytic_gradients, rtol = rtol, atol = atol)
        if not check:
            diff = abs(numerical_gradients - analytic_gradients)
            print('\nAbs Diff:')
            print(diff)
            print('Rel Diff:')
            print(diff / abs(analytic_gradients))
            print('\n')
        num_successes += check
    print('    %i successes in %i runs (%3.1f%%)\n' %(num_successes, num_runs, 100 * num_successes / num_runs))