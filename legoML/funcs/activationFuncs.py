from numpy import atleast_2d, atleast_3d, diag, exp, tanh, zeros


def linearActivations(signals___array, *args, **kwargs):
    return signals___array


def linear_dActivations_over_dSignals(*args, **kwargs):
    return 1.


def logisticActivations(signals___array, *args, **kwargs):
    return (1. / (1. + exp(-signals___array)))


def logistic_dActivations_over_dSignals(signals___array = None, activations___array = None, *args, **kwargs):
    if activations___array is None:
        activations___array = logisticActivations(signals___array)
    return activations___array * (1. - activations___array)


def tanhActivations(signals___array, *args, **kwargs):
    return tanh(signals___array)


def tanh_dActivations_over_dSignals(signals___array = None, activations___array = None, *args, **kwargs):
    if activations___array is None:
        activations___array = tanhActivations(signals___array)
    return 1. - activations___array ** 2


def softmaxActivations(signals___matrixCasesInRows, *args, **kwargs):
    z = signals___matrixCasesInRows - signals___matrixCasesInRows.max(1, keepdims = True)
    expMatrix = exp(z)
    return expMatrix / expMatrix.sum(1, keepdims = True)


def softmax_dActivations_over_dSignals(signals___matrixCasesInRows = None, activations___matrixCasesInRows = None,
                                       *args, **kwargs):
    if activations___matrixCasesInRows is None:
        activations___matrixCasesInRows = softmaxActivations(signals___matrixCasesInRows)
    m, n = activations___matrixCasesInRows.shape
    d = zeros([m, n, n])
    for i in range(m):
        a = activations___matrixCasesInRows[i]
        a_2d = atleast_2d(a)
        d[i] = atleast_3d(diag(a) - (a_2d.T).dot(a_2d)).transpose(2, 0, 1)
    return d


def activations(signals___array, nameOfActivationFunc = 'linear', *args, **kwargs):
    funcs___dict = {'linear': lambda s: linearActivations(s),
                    'logistic': lambda s: logisticActivations(s),
                    'tanh': lambda s: tanhActivations(s),
                    'softmax': lambda s: softmaxActivations(s)}
    return funcs___dict[nameOfActivationFunc](signals___array)


def dActivations_over_dSignals(signals___array = None, activations___array = None, nameOfActivationFunc = 'linear',
                               *args, **kwargs):
    funcs___dict = {'linear': lambda s, a: linearActivations(s, a),
                    'logistic': lambda s, a: logisticActivations(s, a),
                    'tanh': lambda s, a: tanhActivations(s, a),
                    'softmax': lambda s, a: softmaxActivations(s, a)}
    return funcs___dict[nameOfActivationFunc](signals___array, activations___array)