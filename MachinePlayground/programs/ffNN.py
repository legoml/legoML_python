from numpy.random import *
from MachinePlayground.Classes import Piece, Process, Program, connectProcesses
from MachinePlayground.pieces.pieces_zzzCommonFuncs import *
from MachinePlayground.pieces.pieces_costFuncs import *



def ffNN(numsNodes, activationFuncs, addBiases = [True], *args, **kwargs):

    # VARS
    vars = ('weightsVector', 'weights', 'signals', 'activations', 'hypoOutputs', 'targetOutputs', 'cost_withoutRegul')

    pieces = {}
    processes = {}
    numLayers = len(numsNodes)
    for l in range(len(addBiases) + 1, numLayers - 1):
        addBiases += addBiases[-1]
    weightsShapes___list = []
    for l in range(numLayers - 1):
        weightsShapes___list[l] = [numsNodes[l] + addBiases[l], numsNodes[l + 1]]

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
             'cost_crossEntropy_binaryClasses_averageOverCases': 'cost_withoutRegul'}),
        'softmax': piece_crossEntropy_multiClasses_averageOverCases().install(
            {'targetOutputs___matrixRowsForCases': 'targetOutputs',
             'hypoOutputs___matrixRowsForCases': 'hypoOutputs',
             'cost_crossEntropy_multiClasses_averageOverCases': 'cost_withoutRegul'})}

    pieces['cost_withoutRegul'] = dictOfPieces_forCostFuncs[activationFuncs[numLayers - 1]]

    pieces['dCostWithoutRegul_over_dSignalToTopLayer'] = Piece(
        forwards = {},
        backwards = {('dOverD', 'cost_withoutRegul', ('signals', numLayers - 2)):
                        [lambda t, h: h - t,
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
                                         ['cost_withoutRegul', ('signals', numLayers - 2)]])
    for l in reversed(range(numLayers - 1)):
        processes['backwardPass'].addSteps(
            [pieces[('signals', l)], None, ['cost_withoutRegul', ('weights', l)]])
        if l > 0:
            processes['backwardPass'].addSteps(
                [pieces[('signals', l)], None, ['cost_withoutRegul', ('activations', l)]],
                [pieces[('activations', l)], None, ['cost_withoutRegul', ('signals', l - 1)]])
    processes['backwardPass'].addSteps(
        [pieces['weights_betweenVectorAndDict'], None, ['cost_withoutRegul', 'weightsVector']])

    # PROCESS: cost_and_dCost_over_dWeights_withoutRegul
    processes['cost_and_dCost_over_dWeights_withoutRegul'] = connectProcesses(
        processes['forwardPass'], processes['cost_withoutRegul'], processes['backwardPass'])

    return Program(vars, pieces, processes)