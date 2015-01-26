from numpy import exp, tanh
from MachinePlayground.Classes import Piece
from MachinePlayground.funcs.zzzCommonFuncs import addBiasElements, deleteBiasElements,\
    fromVector_toArrays_inDictListTuple, fromArrays_inDictListTuple_toVector,\
    softmax, softmax_dOutputs_over_dInputs, softmax_dOverDInputs_from_dOverDOutputs



def piece_equal():
    forwards = {'outputs___array':
                    [lambda inp: inp,
                     {'inp': 'inputs___array'}]}
    backwards = {('dOverD', 'inputs___array'):
                     [lambda ddOutp: ddOutp,
                      {'ddOutp': ('dOverD', 'outputs___array')}]}
    return Piece(forwards, backwards)



def piece_fromVector_toArrays_inDictListTuple(shapes___list, type = 'dict'):

    forwards = {'arrays___inDictListTuple':
                    [lambda v: fromVector_toArrays_inDictListTuple(v, shapes___list, type),
                     {'v': 'vector'}]}

    backwards = {('dOverD', 'vector'):
                    [lambda ddA: fromArrays_inDictListTuple_toVector(ddA),
                     {'ddA': ('dOverD', 'arrays___inDictListTuple')}]}

    return Piece(forwards, backwards)



def piece_multiplyMatrices_ofInputsAndWeights(addBiasColumnToInputs = True, *args, **kwargs):

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