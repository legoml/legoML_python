from numpy import allclose
from numpy.random import *
from MachinePlayground.Classes import *
from MachinePlayground.funcs.zzzMiscFuncs import *
from MachinePlayground.funcs.zzzCommonFuncs import *
from MachinePlayground.funcs.activationFuncs import *



#def group_ffNN(*args, **kwargs):



#    return Group()







#    return


#def process_computeCost:
#    return

#def process_backwardPass:
#    return





class classFFNN:
    def __init__(self, inputDimSizes_perCase___vector = [1])
        self.inputDimSizes_perCase = inputDimSizes_perCase___vector;
        self.numLayers = numLayers;
        self.transformFuncs = [];
        self.weightDimSizes = {};
        self.weights = {};
        self.numWeights = 0;
        self.costFuncType = '';


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