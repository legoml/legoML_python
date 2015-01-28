from numpy import array, atleast_2d
from numpy.random import *
from MachinePlayground.Classes import Piece, Process, Program, connectProcesses
from MachinePlayground.pieces.pieces_zzzCommonFuncs import *
from MachinePlayground.pieces.pieces_costFuncs import *
from MachinePlayground.pieces.pieces_regulFuncs import *



def program_ffNN(numsNodes, activationFuncs, addBiases = [True], *args, **kwargs):

    pieces = {}
    processes = {}
    numLayers = len(numsNodes)
    for l in range(len(addBiases) + 1, numLayers - 1):
        addBiases += addBiases[-1]
    weightsShapes___list = []
    numWeights = 0
    for l in range(numLayers - 1):
        shape = array([numsNodes[l] + addBiases[l], numsNodes[l + 1]])
        weightsShapes___list += [shape]
        numWeights += shape.prod()

    # VARS
    vars = {'weightsVector': array(numWeights * [0]),
            'weights': {},
            'inputs': array([]),
            'signals': {},
            'activations': {},
            'hypoOutputs': array([]),
            'targetOutputs': array([]),
            'cost_withoutRegul': array([]),
            'cost_weightPenalty': array([])}
    if activationFuncs[-1] == 'linear':
        vars['rootMeanSquareError'] = array([])
    elif activationFuncs[-1] == 'logistic':
        vars['percentConfidence'] = array([])
        vars['positiveClassSkewnesses'] = atleast_2d(array([1]))
    elif activationFuncs[-1] == 'softmax':
        vars['percentConfidence'] = array([])
        vars['classSkewnesses'] = atleast_2d(array(numsNodes[-1] * [1]))

    # PIECES
    pieces['weights_betweenVectorAndDict'] = piece_fromVector_toArrays_inDictListTuple(weightsShapes___list).install(
        {'vector': 'weightsVector',
         'arrays___inDictListTuple': 'weights'})

    pieces[('activations', 0)] = piece_equal().install({'inputs___array': 'inputs',
                                                        'outputs___array': ('activations', 0)})

    dictOfPieces_forActivationFuncs = {
        'linear': piece_linear().install({'inputs___array': 'signal',
                                          'outputs___array': 'activation'}),
        'logistic': piece_logistic().install({'inputs___array': 'signal',
                                              'outputs___array': 'activation'}),
        'tanh': piece_tanh().install({'inputs___array': 'signal',
                                      'outputs___array': 'activation'}),
        'softmax': piece_softmax().install({'inputs___matrixRowsForCases': 'signal',
                                            'outputs___matrixRowsForCases': 'activation'})}

    for l in range(1, numLayers):
        pieces[('signals', l - 1)] = piece_multiplyMatrices_ofInputsAndWeights(addBiases[l - 1]).install(
            {'inputs___matrixRowsForCases': ('activations', l - 1),
             'weights___matrix': ('weights', l - 1),
             'outputs___matrixRowsForCases': ('signals', l - 1)})
        pieces[('activations', l)] = dictOfPieces_forActivationFuncs[activationFuncs[l - 1]].install(
            {'signal': ('signals', l - 1),
             'activation': ('activations', l)})

    pieces['hypoOutputs'] = piece_equal().install({'inputs___array': ('activations', numLayers - 1),
                                                   'outputs___array': 'hypoOutputs'})

    dictOfPieces_forCostFuncs = {
        'linear': piece_squareError_half_averageOverCases().install(
            {'targetOutputs___array': 'targetOutputs',
             'hypoOutputs___array': 'hypoOutputs',
             'cost_squareError_half_averageOverCases': 'cost_withoutRegul'}),
        'logistic': piece_crossEntropy_binaryClasses_averageOverCases().install(
            {'targetOutputs___array': 'targetOutputs',
             'hypoOutputs___array': 'hypoOutputs',
             'positiveClassSkewnesses': 'positiveClassSkewnesses',
             'cost_crossEntropy_binaryClasses_averageOverCases': 'cost_withoutRegul'}),
        'softmax': piece_crossEntropy_multiClasses_averageOverCases().install(
            {'targetOutputs___matrixRowsForCases': 'targetOutputs',
             'hypoOutputs___matrixRowsForCases': 'hypoOutputs',
             'classSkewnesses': 'classSkewnesses',
             'cost_crossEntropy_multiClasses_averageOverCases': 'cost_withoutRegul'})}

    pieces['cost_withoutRegul'] = dictOfPieces_forCostFuncs[activationFuncs[numLayers - 2]]

    pieces['dCostWithoutRegul_over_dSignalToTopLayer'] = Piece(
        forwards = {},
        backwards = {('dOverD', 'cost_withoutRegul', ('signals', numLayers - 2)):
                        [lambda t, h: (h - t) / t.shape[0],
                         {'t': 'targetOutputs',
                          'h': 'hypoOutputs'}]})

    # PROCESS: forwardPass
    processes['forwardPass'] = Process(pieces['weights_betweenVectorAndDict'], pieces[('activations', 0)])
    for l in range(1, numLayers):
        processes['forwardPass'].addSteps(pieces[('signals', l - 1)], pieces[('activations', l)])
    processes['forwardPass'].addSteps(pieces[('hypoOutputs')])

    # PROCESS: cost_withoutRegul
    processes['cost_withoutRegul'] = Process(pieces['cost_withoutRegul'])

    # PROCESS: backwardPass
    processes['backwardPass'] = Process([pieces[('dCostWithoutRegul_over_dSignalToTopLayer')], None,
                                         ['cost_withoutRegul', [('signals', numLayers - 2)]]])
    for l in reversed(range(numLayers - 1)):
        processes['backwardPass'].addSteps(
            [pieces[('signals', l)], None, ['cost_withoutRegul', [('weights', l)]]])
        if l > 0:
            processes['backwardPass'].addSteps(
                [pieces[('signals', l)], None, ['cost_withoutRegul', [('activations', l)]]],
                [pieces[('activations', l)], None, ['cost_withoutRegul', [('signals', l - 1)]]])
    processes['backwardPass'].addSteps(
        [pieces['weights_betweenVectorAndDict'], None, ['cost_withoutRegul', ['weightsVector']]])

    # PROCESS: cost_and_dCost_over_dWeights_withoutRegul
    processes['cost_and_dCost_over_dWeights_withoutRegul'] = connectProcesses(
        processes['forwardPass'], processes['cost_withoutRegul'], processes['backwardPass'])

    # PROCESS: weight update: gradient

    return Program(vars, pieces, processes)