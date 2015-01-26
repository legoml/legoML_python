from numpy import allclose
from numpy.random import *
from copy import deepcopy
from MachinePlayground._common import *
from MachinePlayground.programs import ffNN



def TEST_ffNN_checkGradients(numRuns = 1000, rtol = 1.e-6, atol = 1.e-6):

    def cost(model, weightsVector):
        p0 = deepcopy(model)
        p0.vars['weightsVector'] = weightsVector
        p0.run('forwardPass')
        p0.run('cost_withoutRegul')
        return p0.vars['cost_withoutRegul']

    maxNumCases = 5
    maxNumLayers = 5
    maxNumNodes_perLayer = 5

    print('\nTEST ffNN (check gradients):')
    L = randint(1, maxNumLayers) + 1
    numsNodes = []
    activationFuncs = []
    addBiases = []
    for l in range(L):
        numsNodes += [randint(maxNumNodes_perLayer) + 1]
        if l < L - 1:
            if l < L - 2:
                activationFuncs += [('linear', 'logistic', 'tanh', 'softmax')[randint(4)]]
            else:
                activationFuncs += [('linear', 'logistic', 'softmax')[randint(3)]]
            addBiases += [randint(2)]
    p = ffNN.ffNN(numsNodes, activationFuncs, addBiases)

    numSuccesses = 0
    for r in range(numRuns):
        m = randint(maxNumCases) + 1
        p.vars['inputs'] = rand(m, numsNodes[0])
        if activationFuncs[-1] == 'softmax':
            y = rand(m, numsNodes[-1])
            yMax = y.max(1, keepdims = True)
            p.vars['targetOutputs'] = 1. * (y == yMax)
        else:
            p.vars['targetOutputs'] = rand(m, numsNodes[-1])
        weightsVector = rand(*p.vars['weightsVector'].shape)
        p.vars['weightsVector'] = weightsVector
        p.run('cost_and_dCost_over_dWeights_withoutRegul')
        analyticGrads = p.vars[('dOverD', 'cost_withoutRegul', 'weightsVector')]
        approxGrads = approxGradients(lambda w: cost(p, w), weightsVector)
        numSuccesses += allclose(approxGrads, analyticGrads, rtol = rtol, atol = atol)
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))