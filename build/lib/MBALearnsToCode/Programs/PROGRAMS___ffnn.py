from MBALearnsToCode import Process, Program, connect_processes
from MBALearnsToCode.Pieces.PIECES___CommonFunctions import *
from MBALearnsToCode.Pieces import *



def PROGRAM___ffnn(nums_nodes, activation_functions, add_biases = [True]):

    pieces = {}
    processes = {}

    num_layers = len(nums_nodes)

    for l in range(len(add_biases) + 1, num_layers - 1):
        add_biases += add_biases[-1]
    weights_shapes___list = []
    num_weights = 0
    for l in range(num_layers - 1):
        shape = array([nums_nodes[l] + add_biases[l], nums_nodes[l + 1]])
        weights_shapes___list += [shape]
        num_weights += shape.prod()

    # PIECES
    pieces['weights_from_vector_to_dict'] = PIECE___from_vector_to_arrays(weights_shapes___list).install(
        change_keys = {'vector': 'weights_vector',
                       'arrays': 'weights'})

    pieces[('activations', 0)] = PIECE___equal().install(
        change_keys = {'inputs': 'inputs',
                       'outputs': ('activations', 0)})

    dict_of_pieces_for_activation_functions =\
        {'linear': PIECE___linear(),
         'logistic': PIECE___logistic(),
         'tanh': PIECE___tanh(),
         'softmax': PIECE___softmax()}

    for l in range(1, num_layers):
        pieces[('signals', l - 1)] = PIECE___matrix_product_of_inputs_and_weights(add_biases[l - 1]).install(
            change_keys = {'inputs': ('activations', l - 1),
                           'weights': ('weights', l - 1),
                           'outputs': ('signals', l - 1)})
        pieces[('activations', l)] = dict_of_pieces_for_activation_functions[activation_functions[l - 1]].install(
            change_keys = {'inputs': ('signals', l - 1),
                           'outputs': ('activations', l)})

    pieces['predicted_outputs'] = PIECE___equal().install(
        change_keys = {'inputs': ('activations', num_layers - 1),
                       'outputs': 'predicted_outputs'})

    dict_of_pieces_for_cost_functions =\
        {'linear': PIECE___average_half_square_error().install(
            change_keys = {'target_outputs': 'target_outputs',
                           'predicted_outputs': 'predicted_outputs',
                           'average_half_square_error': 'cost'}),
         'logistic': PIECE___average_binary_class_cross_entropy().install(
             change_keys = {'target_outputs': 'target_outputs',
                            'predicted_outputs': 'predicted_outputs',
                            'average_binary_class_cross_entropy': 'cost'}),
         'softmax': PIECE___average_multi_class_cross_entropy().install(
             change_keys = {'target_outputs': 'target_outputs',
                            'predicted_outputs': 'predicted_outputs',
                            'average_multi_class_cross_entropy': 'cost'})}

    pieces['cost'] = dict_of_pieces_for_cost_functions[activation_functions[num_layers - 2]]

    pieces['d_cost_over_d_signal_to_top_layer'] = Piece(
        forwards = {},
        backwards = {('DOVERD', 'cost', ('signals', num_layers - 2)):
                        [lambda t, h: (h - t) / t.shape[0],
                         {'t': 'target_outputs',
                          'h': 'predicted_outputs'}]})

    # PROCESS: forward_pass
    processes['forward_pass'] = Process(pieces['weights_from_vector_to_dict'], pieces[('activations', 0)])
    for l in range(1, num_layers):
        processes['forward_pass'].add_steps(pieces[('signals', l - 1)], pieces[('activations', l)])
    processes['forward_pass'].add_steps(pieces[('predicted_outputs')])

    # PROCESS: cost
    processes['cost'] = Process(pieces['cost'])

    # PROCESS: backward_pass
    processes['backward_pass'] = Process([pieces[('d_cost_over_d_signal_to_top_layer')], None,
                                         ['cost', [('signals', num_layers - 2)]]])
    for l in reversed(range(num_layers - 1)):
        processes['backward_pass'].add_steps(
            [pieces[('signals', l)], None, ['cost', [('weights', l)]]])
        if l > 0:
            processes['backward_pass'].add_steps(
                [pieces[('signals', l)], None, ['cost', [('activations', l)]]],
                [pieces[('activations', l)], None, ['cost', [('signals', l - 1)]]])
    processes['backward_pass'].add_steps(
        [pieces['weights_from_vector_to_dict'], None, ['cost', ['weights_vector']]])

    # PROCESS: cost_and_d_cost_over_d_weights
    processes['cost_and_d_cost_over_d_weights'] = connect_processes(
        processes['forward_pass'], processes['cost'], processes['backward_pass'])

    return Program(pieces, processes)



def PROGRAM___ffnn_unskewed_classification(nums_nodes, activation_functions, add_biases = [True]):

    pieces = {}
    processes = {}

    num_layers = len(nums_nodes)
    for l in range(len(add_biases) + 1, num_layers - 1):
        add_biases += add_biases[-1]
    weights_shapes___list = []
    num_weights = 0
    for l in range(num_layers - 1):
        shape = array([nums_nodes[l] + add_biases[l], nums_nodes[l + 1]])
        weights_shapes___list += [shape]
        num_weights += shape.prod()

    # PIECES
    pieces['weights_from_vector_to_dict'] = PIECE___from_vector_to_arrays(weights_shapes___list).install(
        change_keys = {'vector': 'weights_vector',
                       'arrays': 'weights'})

    pieces[('activations', 0)] = PIECE___equal().install(
        change_keys = {'inputs': 'inputs',
                       'outputs': ('activations', 0)})

    dict_of_pieces_for_activation_functions =\
        {'linear': PIECE___linear(),
         'logistic': PIECE___logistic(),
         'tanh': PIECE___tanh(),
         'softmax': PIECE___softmax()}

    for l in range(1, num_layers):
        pieces[('signals', l - 1)] = PIECE___matrix_product_of_inputs_and_weights(add_biases[l - 1]).install(
            change_keys = {'inputs': ('activations', l - 1),
                           'weights': ('weights', l - 1),
                           'outputs': ('signals', l - 1)})
        pieces[('activations', l)] = dict_of_pieces_for_activation_functions[activation_functions[l - 1]].install(
            change_keys = {'inputs': ('signals', l - 1),
                           'outputs': ('activations', l)})

    pieces['predicted_outputs'] = PIECE___equal().install(
        change_keys = {'inputs': ('activations', num_layers - 1),
                       'outputs': 'predicted_outputs'})

    dict_of_pieces_for_cost_functions =\
        {'logistic': PIECE___average_unskewed_binary_class_cross_entropy().install(
            change_keys = {'target_outputs': 'target_outputs',
                           'predicted_outputs': 'predicted_outputs',
                           'positive_class_skewnesses': 'positive_class_skewnesses',
                           'average_unskewed_binary_class_cross_entropy': 'cost'}),
         'softmax': PIECE___average_unskewed_multi_class_cross_entropy().install(
             change_keys= {'target_outputs': 'target_outputs',
                           'predicted_outputs': 'predicted_outputs',
                           'multi_class_skewnesses': 'multi_class_skewnesses',
                           'average_unskewed_multi_class_cross_entropy': 'cost'})}

    pieces['cost'] = dict_of_pieces_for_cost_functions[activation_functions[num_layers - 2]]

    pieces['d_cost_over_d_signal_to_top_layer'] = Piece(
        forwards = {},
        backwards = {('DOVERD', 'cost', ('signals', num_layers - 2)):
                        [lambda t, h: (h - t) / t.shape[0],
                         {'t': 'target_outputs',
                          'h': 'predicted_outputs'}]})

    # PROCESS: forward_pass
    processes['forward_pass'] = Process(pieces['weights_from_vector_to_dict'], pieces[('activations', 0)])
    for l in range(1, num_layers):
        processes['forward_pass'].add_steps(pieces[('signals', l - 1)], pieces[('activations', l)])
    processes['forward_pass'].add_steps(pieces[('predicted_outputs')])

    # PROCESS: cost
    processes['cost'] = Process(pieces['cost'])

    # PROCESS: backward_pass
    processes['backward_pass'] = Process([pieces[('d_cost_over_d_signal_to_top_layer')], None,
                                         ['cost', [('signals', num_layers - 2)]]])
    for l in reversed(range(num_layers - 1)):
        processes['backward_pass'].add_steps(
            [pieces[('signals', l)], None, ['cost', [('weights', l)]]])
        if l > 0:
            processes['backward_pass'].add_steps(
                [pieces[('signals', l)], None, ['cost', [('activations', l)]]],
                [pieces[('activations', l)], None, ['cost', [('signals', l - 1)]]])
    processes['backward_pass'].add_steps(
        [pieces['weights_from_vector_to_dict'], None, ['cost', ['weights_vector']]])

    # PROCESS: cost_and_d_cost_over_d_weights
    processes['cost_and_d_cost_over_d_weights'] = connect_processes(
        processes['forward_pass'], processes['cost'], processes['backward_pass'])

    return Program(pieces, processes)