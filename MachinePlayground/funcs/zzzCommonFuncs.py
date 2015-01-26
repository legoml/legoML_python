from numpy import array, atleast_2d, atleast_3d, concatenate, delete, diag, exp, ndarray, ones, squeeze, tanh, zeros



def identity(*args):
    # return argument object(s) itself/themselves
    if len(args) == 1:
        return args[0]
    else:
        return args



def addBiasElements(arrayA, numsBiases_toAdd = [0, 1]):
    a = arrayA.copy()
    for d in range(len(numsBiases_toAdd)):
        numBiases_toAdd = numsBiases_toAdd[d]
        if numBiases_toAdd > 0:
            s = list(a.shape)
            s[d] = numBiases_toAdd
            a = concatenate((ones(s), a), axis = d)
    return a


def deleteBiasElements(arrayA, numsBiases_toDelete = [1]):
    a = arrayA.copy()
    for d in range(len(numsBiases_toDelete)):
        numBiases_toDelete = numsBiases_toDelete[d]
        if numBiases_toDelete > 0:
            a = delete(a, range(numBiases_toDelete), axis = d)
    return a


def zeroBiasElements(arrayA, numsBiases_toZero_upTo3D = [1]):
    a = arrayA.copy()
    for d in range(len(numsBiases_toZero_upTo3D)):
        numBiases_toZero = numsBiases_toZero_upTo3D[d]
        if numBiases_toZero > 0:
            if d == 0:
                a[range(numBiases_toZero)] = 0
            elif d == 1:
                a[:, range(numBiases_toZero)] = 0
            elif d == 2:
                a[:, :, range(numBiases_toZero)] = 0
    return a


def fromArrays_inDictListTuple_toVector(dict_orList_orTuple):
    v = []
    for i in range(len(dict_orList_orTuple)):
        v += dict_orList_orTuple[i].flat
    return array(v)


def fromVector_toArrays_inDictListTuple(vector, shapes___list, type = 'dict'):
    if type == 'dict':
        arrays = {}
    else:
        arrays = len(shapes___list) * [[]]
    v = vector.copy()
    for i, s in enumerate(shapes___list):
        if not isinstance(s, ndarray):
            s = array(s)
        numElements = s.prod()
        arrays[i] = v[:numElements].reshape(s)
        v = v[numElements:]
    if type == 'tuple':
        arrays = tuple(arrays)
    return arrays



def multiplyMatrices_ofInputsAndWeights(inputs___matrixRowsForCases, weights___matrix,
                                        addBiasColumn_toInputs = True):
    return addBiasElements(inputs___matrixRowsForCases, [0, addBiasColumn_toInputs]).dot(weights___matrix)


def multiplyMatrices_dOverDInputs_from_dOverDOutputs(dOverDOutputs___matrixRowsForCases, weights___matrix,
                                                     addBiasColumn_toInputs = True):
    return dOverDOutputs___matrixRowsForCases.dot(
        deleteBiasElements(weights___matrix, [addBiasColumn_toInputs]).T)


def multiplyMatrices_dOverDWeights_from_dOverDOutputs(dOverDOutputs___matrixRowsForCases, inputs___matrixRowsForCases,
                                                      addBiasColumn_toInputs = True):
    return (addBiasElements(inputs___matrixRowsForCases, [0, addBiasColumn_toInputs]).T).dot(
        dOverDOutputs___matrixRowsForCases)



def linear(inputs___array, *args, **kwargs):
    return inputs___array


def linear_dOutputs_over_dInputs(*args, **kwargs):
    return 1.


def linear_dOverDInputs_from_dOverDOutputs(dOverDOutputs, *args, **kwargs):
    return dOverDOutputs



def logistic(inputs___array, *args, **kwargs):
    return (1. / (1. + exp(-inputs___array)))


def logistic_dOutputs_over_dInputs(inputs___array = None, outputs___array = None, *args, **kwargs):
    if outputs___array is None:
        outputs___array = logistic(inputs___array)
    return outputs___array * (1. - outputs___array)


def logistic_dOverDInputs_from_dOverDOutputs(dOverDOutputs, dOutputs_over_dInputs, *args, **kwargs):
    return dOverDOutputs * dOutputs_over_dInputs



def tanH(inputs___array, *args, **kwargs):
    return tanh(inputs___array)


def tanh_dOutputs_over_dInputs(inputs___array = None, outputs___array = None, *args, **kwargs):
    if outputs___array is None:
        outputs___array = tanh(inputs___array)
    return 1. - outputs___array ** 2


def tanh_dOverDInputs_from_dOverDOutputs(dOverDOutputs, dOutputs_over_dInputs, *args, **kwargs):
    return dOverDOutputs * dOutputs_over_dInputs



def softmax(inputs___matrixRowsForCases, *args, **kwargs):
    inp = inputs___matrixRowsForCases - inputs___matrixRowsForCases.max(1, keepdims = True)
    expMatrix = exp(inp)
    return expMatrix / expMatrix.sum(1, keepdims = True)


def softmax_dOutputs_over_dInputs(inputs___matrixRowsForCases = None, outputs___matrixRowsForCases = None,
                                  *args, **kwargs):
    if outputs___matrixRowsForCases is None:
        outputs___matrixRowsForCases = softmax(inputs___matrixRowsForCases)
    m, n = outputs___matrixRowsForCases.shape
    d = zeros([m, n, n])
    for i in range(m):
        outp = outputs___matrixRowsForCases[i]
        outp_2d = atleast_2d(outp)
        d[i] = atleast_3d(diag(outp) - (outp_2d.T).dot(outp_2d)).transpose(2, 0, 1)
    return d


def softmax_dOverDInputs_from_dOverDOutputs(dOverDOutputs, dOutputs_over_dInputs, *args, **kwargs):
    return squeeze(((atleast_3d(dOverDOutputs).repeat(dOutputs_over_dInputs.shape[2], axis = 2)
             * dOutputs_over_dInputs).sum(1, keepdims = True)).transpose(0, 2, 1), axis = 2)