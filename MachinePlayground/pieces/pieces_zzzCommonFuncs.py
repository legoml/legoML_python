from numpy import array, atleast_2d, atleast_3d, concatenate, delete, diag, exp, ndarray, ones, squeeze, tanh, zeros
from MachinePlayground.Classes import Piece



def piece_equal():
    forwards = {'outputs___array':
                    [lambda inp: inp,
                     {'inp': 'inputs___array'}]}
    backwards = {('dOverD', 'inputs___array'):
                     [lambda ddOutp: ddOutp,
                      {'ddOutp': ('dOverD', 'outputs___array')}]}
    return Piece(forwards, backwards)



def piece_fromVector_toArrays_inDictListTuple(shapes___list, type = 'dict'):

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

    forwards = {'arrays___inDictListTuple':
                    [lambda v: fromVector_toArrays_inDictListTuple(v, shapes___list, type),
                     {'v': 'vector'}]}

    backwards = {('dOverD', 'vector'):
                    [lambda ddA: fromArrays_inDictListTuple_toVector(ddA),
                     {'ddA': ('dOverD', 'arrays___inDictListTuple')}]}

    return Piece(forwards, backwards)



def piece_multiplyMatrices_ofInputsAndWeights(addBiasColumnToInputs = True, *args, **kwargs):

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

    forwards = {'outputs___matrixRowsForCases':
                    [lambda inp, w: addBiasElements(inp, [0, addBiasColumnToInputs]).dot(w),
                     {'inp': 'inputs___matrixRowsForCases',
                      'w': 'weights___matrix'}]}

    backwards = {('dOverD', 'inputs___matrixRowsForCases'):
                    [lambda ddOutp, w: ddOutp.dot(deleteBiasElements(w, [addBiasColumnToInputs]).T),
                     {'ddOutp': ('dOverD', 'outputs___matrixRowsForCases'),
                      'w': 'weights___matrix'}],

                 ('dOverD', 'weights___matrix'):
                    [lambda ddOutp, inp: (addBiasElements(inp, [0, addBiasColumnToInputs]).T).dot(ddOutp),
                     {'ddOutp': ('dOverD', 'outputs___matrixRowsForCases'),
                      'inp': 'inputs___matrixRowsForCases'}]}

    return Piece(forwards, backwards)



def piece_linear(*args, **kwargs):

    forwards = {'outputs___array':
                    [lambda inp: inp,
                     {'inp': 'inputs___array'}]}

    backwards = {('dOverD', 'inputs___array'):
                     [lambda ddOutp: ddOutp,
                      {'ddOutp': ('dOverD', 'outputs___array')}]}

    return Piece(forwards, backwards)



def piece_logistic(*args, **kwargs):

    forwards = {'outputs___array':
                    [lambda inp: 1. / (1. + exp(-inp)),
                     {'inp': 'inputs___array'}]}

    backwards = {('dOverD', 'inputs___array'):
                     [lambda ddOutp, outp: ddOutp * outp * (1. - outp),
                      {'ddOutp': ('dOverD', 'outputs___array'),
                       'outp': 'outputs___array'}]}

    return Piece(forwards, backwards)



def piece_tanh(*args, **kwargs):

    forwards = {'outputs___array':
                    [lambda inp: tanh(inp),
                     {'inp': 'inputs___array'}]}

    backwards = {('dOverD', 'inputs___array'):
                     [lambda ddOutp, outp: ddOutp * (1. - outp ** 2),
                      {'ddOutp': ('dOverD', 'outputs___array'),
                       'outp': 'outputs___array'}]}

    return Piece(forwards, backwards)



def piece_softmax(*args, **kwargs):

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

    forwards = {'outputs___matrixRowsForCases':
                    [lambda inp: softmax(inp),
                     {'inp': 'inputs___matrixRowsForCases'}]}

    backwards = {('dOverD', 'inputs___matrixRowsForCases'):
                     [lambda ddOutp, outp:
                            softmax_dOverDInputs_from_dOverDOutputs(ddOutp,
                                softmax_dOutputs_over_dInputs(outputs___matrixRowsForCases = outp)),
                      {'ddOutp': ('dOverD', 'outputs___matrixRowsForCases'),
                       'outp': 'outputs___matrixRowsForCases'}]}

    return Piece(forwards, backwards)