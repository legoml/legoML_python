from numpy import array, atleast_2d, atleast_3d, concatenate, delete, diag, exp, ndarray, ones, squeeze, tanh, zeros
from MachinePlayground.Classes import Piece



def PIECE___equal():
    """PIECE___equal

    OUTPUTS multi-dimensional array = INPUTS multi-dimensional array
    """

    forwards = {'outputs':
                    [lambda inp: inp,
                     {'inp': 'inputs'}]}
    backwards = {('DOVERD', 'input'):
                     [lambda ddoutp: ddoutp,
                      {'ddoutp': ('DOVERD', 'outputs')}]}
    return Piece(forwards, backwards)



def PIECE___from_vector_to_arrays(shapes___list, type = 'dict'):
    """PIECE___from_vector_to_arrays

    Converts a vector to dict/list/tuple of multi-dimensional arrays according to a list of shapes
    """

    def from_arrays_to_vector(dict_or_list_or_tuple):
        v = []
        for i in range(len(dict_or_list_or_tuple)):
            v += dict_or_list_or_tuple[i].flat
        return array(v)

    def from_vector_to_arrays(vector, shapes___list, type = 'dict'):
        if type == 'dict':
            arrays = {}
        else:
            arrays = len(shapes___list) * [[]]
        v = vector.copy()
        for i, s in enumerate(shapes___list):
            if not isinstance(s, ndarray):
                s = array(s)
            num_elements = s.prod()
            arrays[i] = v[:num_elements].reshape(s)
            v = v[num_elements:]
        if type == 'tuple':
            arrays = tuple(arrays)
        return arrays

    forwards = {'arrays':
                    [lambda v: from_vector_to_arrays(v, shapes___list, type),
                     {'v': 'vector',
                      'type': 'type'}]}

    backwards = {('DOVERD', 'vector'):
                    [lambda dda: from_arrays_to_vector(dda),
                     {'dda': ('DOVERD', 'arrays')}]}

    return Piece(forwards, backwards)



def PIECE___matrix_product_of_inputs_and_weights(add_bias_column_to_inputs = True):
    """PIECE___multiply_matrices_of_inputs_and_weights

    OUTPUTS matrix (cases in rows) = matrix product of INPUTS matrix (cases in rows) and WEIGHTS matrix
    """

    def add_bias_elements(array_a, nums_biases_to_add = [0, 1]):
        a = array_a.copy()
        for d in range(len(nums_biases_to_add)):
            num_biases_to_add = nums_biases_to_add[d]
            if num_biases_to_add > 0:
                s = list(a.shape)
                s[d] = num_biases_to_add
                a = concatenate((ones(s), a), axis = d)
        return a

    def delete_bias_elements(array_a, nums_biases_to_delete = [1]):
        a = array_a.copy()
        for d in range(len(nums_biases_to_delete)):
            num_biases_to_delete = nums_biases_to_delete[d]
            if num_biases_to_delete > 0:
                a = delete(a, range(num_biases_to_delete), axis = d)
        return a

    forwards = {'outputs':
                    [lambda inp, w: add_bias_elements(inp, [0, add_bias_column_to_inputs]).dot(w),
                     {'inp': 'inputs',
                      'w': 'weights'}]}

    backwards = {('DOVERD', 'inputs'):
                    [lambda ddoutp, w: ddoutp.dot(delete_bias_elements(w, [add_bias_column_to_inputs]).T),
                     {'ddoutp': ('DOVERD', 'outputs'),
                      'w': 'weights'}],

                 ('DOVERD', 'weights'):
                    [lambda ddoutp, inp: (add_bias_elements(inp, [0, add_bias_column_to_inputs]).T).dot(ddoutp),
                     {'ddoutp': ('DOVERD', 'outputs'),
                      'inp': 'inputs'}]}

    return Piece(forwards, backwards)



def PIECE___linear():
    """PIECE___linear

    OUTPUTS multi-dimensional array = INPUTS multi-dimensional array
    """

    forwards = {'outputs':
                    [lambda inp: inp,
                     {'inp': 'inputs'}]}

    backwards = {('DOVERD', 'inputs'):
                     [lambda ddoutp: ddoutp,
                      {'ddoutp': ('DOVERD', 'outputs')}]}

    return Piece(forwards, backwards)



def PIECE___logistic():
    """PIECE___logistic

    OUTPUTS multi-dimensional array = logistic function of INPUTS multi-dimensional array
    """

    forwards = {'outputs':
                    [lambda inp: 1. / (1. + exp(-inp)),
                     {'inp': 'inputs'}]}

    backwards = {('DOVERD', 'inputs'):
                     [lambda ddoutp, outp: ddoutp * outp * (1. - outp),
                      {'ddoutp': ('DOVERD', 'outputs'),
                       'outp': 'outputs'}]}

    return Piece(forwards, backwards)



def PIECE___tanh():
    """PIECE___tanh

    OUTPUTS multi-dimensional array = hyperbolic tangent function of INPUTS multi-dimensional array
    """

    forwards = {'outputs':
                    [lambda inp: tanh(inp),
                     {'inp': 'inputs'}]}

    backwards = {('DOVERD', 'inputs'):
                     [lambda ddoutp, outp: ddoutp * (1. - outp ** 2),
                      {'ddoutp': ('DOVERD', 'outputs'),
                       'outp': 'outputs'}]}

    return Piece(forwards, backwards)



def PIECE___softmax():
    """PIECE___softmax

    OUTPUTS matrix (cases in rows) = softmax function of INPUTS matrix (cases in rows)
    """

    def softmax(inputs):
        inp = inputs - inputs.max(1, keepdims = True)
        exp_matrix = exp(inp)
        return exp_matrix / exp_matrix.sum(1, keepdims = True)

    def softmax_d_outputs_over_d_inputs(inputs = None, outputs = None):
        if outputs is None:
            outputs = softmax(inputs)
        m, n = outputs.shape
        d = zeros([m, n, n])
        for i in range(m):
            outp = outputs[i]
            outp_2d = atleast_2d(outp)
            d[i] = atleast_3d(diag(outp) - (outp_2d.T).dot(outp_2d)).transpose(2, 0, 1)
        return d

    def softmax_doverd_inputs_from_doverd_outputs(doverd_outputs, d_outputs_over_d_inputs):
        return squeeze(((atleast_3d(doverd_outputs).repeat(d_outputs_over_d_inputs.shape[2], axis = 2)
                * d_outputs_over_d_inputs).sum(1, keepdims = True)).transpose(0, 2, 1), axis = 2)

    forwards = {'outputs':
                    [lambda inp: softmax(inp),
                     {'inp': 'inputs'}]}

    backwards = {('DOVERD', 'inputs'):
                     [lambda ddoutp, outp:
                            softmax_doverd_inputs_from_doverd_outputs(ddoutp,
                                softmax_d_outputs_over_d_inputs(outputs = outp)),
                      {'ddoutp': ('DOVERD', 'outputs'),
                       'outp': 'outputs'}]}

    return Piece(forwards, backwards)