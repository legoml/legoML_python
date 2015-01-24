from numpy import allclose
from numpy.random import *
from MachinePlayground.Classes import Piece, Process, Program
from MachinePlayground.pieces.pieces_zzzCommonFuncs import *
from MachinePlayground.pieces.pieces_costFuncs import *
from MachinePlayground.funcs.activationFuncs import *


class classFFNN:
    def __init__(self, inputDimSizes_perCase___vector = [1])


def ffNN(numsNodes, activationFuncs, *args, **kwargs):
    numLayers = len(numsNodes)
    pieces = {}
    processes = {}

    pieces[('activations', 0)] = piece_equal().install({'inputs___array': 'inputs',
                                                        'outputs___array': ('activations', 0)})

    dictOfPieces_forActivationFuncs = {
        'linear': piece_linear().install({'inputs___array': 'signal', 'outputs___array': 'activation'}),
        'logistic': piece_logistic().install({'inputs___array': 'signal', 'outputs___array': 'activation'}),
        'tanh': piece_tanh().install({'inputs___array': 'signal', 'outputs___array': 'activation'}),
        'softmax': piece_softmax().install({'inputs___matrixRowsForCases': 'signal',
                                            'outputs___matrixRowsForCases': 'activation'})}

    for l in range(1, numLayers):
        pieces[('signals', l - 1)] = piece_multiplyMatrices_ofInputsAndWeights(True).install(
            {'inputs___matrixRowsForCases': ('activations', l - 1),
             'weights___matrix': ('weights', l - 1)})
        pieces[('layer', l)] = dictOfPieces_forActivationFuncs[activationFuncs[l - 1]].install(
            {'signal': ('signals', l - 1), 'activation': ('activations', l)})

    pieces['hypoOutputs'] = piece_equal().install({'inputs___array': ('activations', numLayers),
                                                   'outputs___array': 'hypoOutputs'})

    dictOfPieces_forCostFuncs = {
        'linear': piece_squareError_half().install({'targetOutputs___array': 'targetOutputs',
                                                    'hypoOutputs___array': 'hypoOutputs'}),
        'logistic': piece_crossEntropy_binaryClasses().install({'inputs___array': 'signal', 'outputs___array': 'activation'}),
        'softmax': piece_softmax().install({'inputs___matrixRowsForCases': 'signal',
                                            'outputs___matrixRowsForCases': 'activation'})}

    pieces['cost_withoutRegul'] = dictOfPieces_forCostFuncs[activationFuncs[numLayers - 1]]

    (forwards = {('activations', 0): [lambda X: X, {'X': 'inputs'}],
                                       backwards = {'dOverD'}
    )



    self.inputDimSizes_perCase = inputDimSizes_perCase___vector;
    self.numLayers = numLayers;
    self.transformFuncs = [];
    self.weightDimSizes = {};
    self.weights = {};
    self.numWeights = 0;
    self.costFuncType = '';

    return Program(pieces, processes)

def computeCost:
    return

def process_backwardPass:
    return







        for i in range(1, numLayers):
            i += 1
            f0 = [{'activations', i}, ]
            f1 =
            p[i] = Piece(f0, f1, f2)


    def forward_backward(self, inputs___array, targetOutputs___matrixCasesInRows):

        self.activations[0] = inputs___array

        for l in range(1, self.numLayers):

            self.signals[l] =
#


   if iscell...
      (addlLayersNumsNodes_vec_OR_weightDimSizes_list)
      weightDimSizes = ...
         addlLayersNumsNodes_vec_OR_weightDimSizes_list;
      numTransforms = length(weightDimSizes);
   else
      if isempty...
         (addlLayersNumsNodes_vec_OR_weightDimSizes_list)
         addlLayersNumsNodes_vec_OR_weightDimSizes_list = 1;
      endif
      numTransforms = ...
         length(addlLayersNumsNodes_vec_OR_weightDimSizes_list);
      numsNodes = [inputDimSizes_perCase_vec ...
         addlLayersNumsNodes_vec_OR_weightDimSizes_list];
   endif

   transformFuncs = transformFuncs_list;
   numTransformFuncs_specified = length(transformFuncs_list);
   for (l = ...
      (numTransformFuncs_specified + 1) : (numTransforms - 1))
      transformFuncs{l} = 'tanh';
   endfor
   if (length(transformFuncs) == (numTransforms - 1))
      transformFuncs(numTransforms) = 'logistic';
   endif

   if ~iscell...
      (addlLayersNumsNodes_vec_OR_weightDimSizes_list)
      for (l = 1 : numTransforms)
         if strcmp(class(transformFuncs{l}), 'char')
            transformFuncs{l} = ...
               convertText_toTransformFunc(transformFuncs{l});
         endif
         weightDimSizes{l} = ...
            [(numsNodes(l) + transformFuncs{l}.addBias) ...
            numsNodes(l + 1)];
      endfor
   endif

   if (numTransformFuncs_specified < numTransforms) && ...
      (weightDimSizes{numTransforms} > 2)
      if (transformFuncs{numTransforms}.addBias)
         transformFuncs{numTransforms} = 'softmax';
      else
         transformFuncs{numTransforms} = 'softmaxNoBias';
      endif
   endif

   for (l = 1 : numTransforms)
      c = addLayer(c, weightDimSizes{l}, transformFuncs{l});
   endfor

   if (initWeights_rand)
      c = initWeights(c, sigma_or_epsilon, distrib);
   endif

   if (displayOverview)
      overview(c);
   endif

    def addLayer(self, )

    def gradientCheck():

    maxNumNodesPerLayer = 6
    maxNumLayers = 6
    maxNumCases = 9

    m = randint(maxNumCases) + 1
    nI = randint(maxNumNodesPerLayer) + 1
    nL = randint(maxNumLayers) + 1

    ffNN = {}
    ffNN['inputs'] = rand(m, nI)


    for l in range(nL):
        ffNN['weights'][l] = []

    ffNN['weights']



class Lambdas_forFuncs_inFFNN:
    def __init__(self, nameOfActivationFunc = 'linear', addBias = True):
        self.funcSignal = signal