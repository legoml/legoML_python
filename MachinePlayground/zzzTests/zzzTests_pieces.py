from numpy.random import *
from MachinePlayground.pieces.pieces_zzzCommonFuncs import *
from MachinePlayground.pieces.pieces_costFuncs import *


def TEST_piece_equal(numRuns = 1000):
    maxNumDims = 3
    maxDimSize = 9
    print('\nTEST piece_equal:')
    p = piece_equal()
    numSuccesses = 0
    for r in range(numRuns):
        numDims = randint(maxNumDims) + 1
        dimSizes = []
        for d in range(numDims):
            dimSizes += [randint(maxDimSize) + 1]
        inp = rand(*dimSizes)
        numSuccesses += p.checkGradients({'inputs___array': inp})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_multiplyMatrices_ofInputsAndWeights(numRuns = 1000):
    maxNumCases = 9
    maxInputDimSize = 9
    maxOutputDimSize = 9
    print('\nTEST piece_multiplyMatrices_ofInputsAndWeights:')
    numSuccesses = 0
    for r in range(numRuns):
        addBias = randint(2)
        p = piece_multiplyMatrices_ofInputsAndWeights(addBias)
        m = randint(maxNumCases) + 1
        nI = randint(maxInputDimSize) + 1
        nO = randint(maxOutputDimSize) + 1
        inp = rand(m, nI)
        w = rand(nI + addBias, nO)
        numSuccesses += p.checkGradients({'inputs___matrixRowsForCases': inp, 'weights___matrix': w})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_linear(numRuns = 1000):
    maxNumDims = 3
    maxDimSize = 9
    print('\nTEST piece_linear:')
    p = piece_linear()
    numSuccesses = 0
    for r in range(numRuns):
        numDims = randint(maxNumDims) + 1
        dimSizes = []
        for d in range(numDims):
            dimSizes += [randint(maxDimSize) + 1]
        inp = rand(*dimSizes)
        numSuccesses += p.checkGradients({'inputs___array': inp})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_logistic(numRuns = 1000):
    maxNumDims = 3
    maxDimSize = 9
    print('\nTEST piece_logistic:')
    p = piece_logistic()
    numSuccesses = 0
    for r in range(numRuns):
        numDims = randint(maxNumDims) + 1
        dimSizes = []
        for d in range(numDims):
            dimSizes += [randint(maxDimSize) + 1]
        inp = rand(*dimSizes)
        numSuccesses += p.checkGradients({'inputs___array': inp})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_tanh(numRuns = 1000):
    maxNumDims = 3
    maxDimSize = 9
    print('\nTEST piece_tanh:')
    p = piece_tanh()
    numSuccesses = 0
    for r in range(numRuns):
        numDims = randint(maxNumDims) + 1
        dimSizes = []
        for d in range(numDims):
            dimSizes += [randint(maxDimSize) + 1]
        inp = rand(*dimSizes)
        numSuccesses += p.checkGradients({'inputs___array': inp})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_softmax(numRuns = 1000):
    maxNumCases = 9
    maxDimSize = 9
    print('\nTEST piece_softmax:')
    p = piece_softmax()
    numSuccesses = 0
    for r in range(numRuns):
        m = randint(maxNumCases) + 1
        n = randint(maxDimSize) + 1
        inp = rand(m, n)
        numSuccesses += p.checkGradients({'inputs___matrixRowsForCases': inp})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_squareError_half_averageOverCases(numRuns = 1000):
    maxNumDims = 3
    maxDimSize = 9
    print('\nTEST piece_squareError_half_averageOverCases:')
    p = piece_squareError_half_averageOverCases()
    numSuccesses = 0
    for r in range(numRuns):
        numDims = randint(maxNumDims) + 1
        dimSizes = []
        for d in range(numDims):
            dimSizes += [randint(maxDimSize) + 1]
        a0 = rand(*dimSizes)
        a1 = rand(*dimSizes)
        numSuccesses += p.checkGradients({'targetOutputs___array': a0, 'hypoOutputs___array': a1})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_crossEntropy_binaryClasses_averageOverCases(numRuns = 1000):
    maxNumDims = 3
    maxDimSize = 9
    print('\nTEST piece_crossEntropy_binaryClasses_averageOverCases:')
    p = piece_squareError_half_averageOverCases()
    numSuccesses = 0
    for r in range(numRuns):
        numDims = randint(maxNumDims) + 1
        dimSizes = []
        for d in range(numDims):
            dimSizes += [randint(maxDimSize) + 1]
        fromArr = rand(*dimSizes)
        ofArr = rand(*dimSizes)
        posSkew = 2 * rand(*dimSizes)
        numSuccesses += p.checkGradients({'targetOutputs___array': fromArr, 'hypoOutputs___array': ofArr,
                                          'positiveClassSkewnesses': posSkew})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))


def TEST_piece_crossEntropy_multiClasses_averageOverCases(numRuns = 1000):
    maxNumCases = 9
    maxDimSize = 9
    print('\nTEST piece_crossEntropy_multiClasses_averageOverCases:')
    p = piece_crossEntropy_multiClasses_averageOverCases()
    numSuccesses = 0
    for r in range(numRuns):
        m = randint(maxNumCases) + 1
        n = randint(maxDimSize) + 1
        fromArr = rand(m, n)
        ofArr = rand(m, n)
        skew = rand(1, n)
        numSuccesses += p.checkGradients({'targetOutputs___matrixRowsForCases': fromArr,
                                          'hypoOutputs___matrixRowsForCases': ofArr,
                                          'classSkewnesses': skew})
    print('   %i successes in %i runs (%3.1f%%)\n' %(numSuccesses, numRuns, 100 * numSuccesses / numRuns))