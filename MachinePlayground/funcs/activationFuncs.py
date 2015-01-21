from numpy import atleast_2d, atleast_3d, diag, exp, tanh, zeros



def linearActivations(signals___array, *args, **kwargs):
    return signals___array


def linear_dActivations_over_dSignals(*args, **kwargs):
    return 1.


def linear_d_over_dSignals_from_d_over_dActivations(d_over_dActivations, dActivations_over_dSignals = None,
                                                    *args, **kwargs):
    return d_over_dActivations



def logisticActivations(signals___array, *args, **kwargs):
    return (1. / (1. + exp(-signals___array)))


def logistic_dActivations_over_dSignals(signals___array = None, activations___array = None, *args, **kwargs):
    if activations___array is None:
        activations___array = logisticActivations(signals___array)
    return activations___array * (1. - activations___array)


def logistic_d_over_dSignals_from_d_over_dActivations(d_over_dActivations, dActivations_over_dSignals, *args, **kwargs):
    return d_over_dActivations * dActivations_over_dSignals



def tanhActivations(signals___array, *args, **kwargs):
    return tanh(signals___array)


def tanh_dActivations_over_dSignals(signals___array = None, activations___array = None, *args, **kwargs):
    if activations___array is None:
        activations___array = tanhActivations(signals___array)
    return 1. - activations___array ** 2


def tanh_d_over_dSignals_from_d_over_dActivations(d_over_dActivations, dActivations_over_dSignals, *args, **kwargs):
    return d_over_dActivations * dActivations_over_dSignals



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


def softmax_d_over_dSignals_from_d_over_dActivations(d_over_dActivations, dActivations_over_dSignals, *args, **kwargs):
    return (((d_over_dActivations.reshape(list(d_over_dActivations.shape) + [1])).repeat(
        dActivations_over_dSignals.shape[2], axis = 3) * dActivations_over_dSignals).sum(1)).transpose(0, 2, 1)



def activations(signals___array, nameOfActivationFunc = 'linear', *args, **kwargs):
    funcs___dict = {'linear': lambda s: linearActivations(s),
                    'logistic': lambda s: logisticActivations(s),
                    'tanh': lambda s: tanhActivations(s),
                    'softmax': lambda s: softmaxActivations(s)}
    return funcs___dict[nameOfActivationFunc](signals___array)


def dActivations_over_dSignals(signals___array = None, activations___array = None, nameOfActivationFunc = 'linear',
                               *args, **kwargs):
    funcs___dict = {'linear': lambda s, a: linear_dActivations_over_dSignals(s, a),
                    'logistic': lambda s, a: logistic_dActivations_over_dSignals(s, a),
                    'tanh': lambda s, a: tanh_dActivations_over_dSignals(s, a),
                    'softmax': lambda s, a: softmax_dActivations_over_dSignals(s, a)}
    return funcs___dict[nameOfActivationFunc](signals___array, activations___array)


def d_over_dSignals_from_d_over_dActivations(d_over_dActivations, dActivations_over_dSignals,
                                             nameOfActivationFunc = 'linear', *args, **kwargs):
    funcs___dict = {'linear': lambda d_dA, dA_dS: linear_d_over_dSignals_from_d_over_dActivations(d_dA, dA_dS),
                    'logistic': lambda d_dA, dA_dS: logistic_d_over_dSignals_from_d_over_dActivations(d_dA, dA_dS),
                    'tanh': lambda d_dA, dA_dS: tanh_d_over_dSignals_from_d_over_dActivations(d_dA, dA_dS),
                    'softmax': lambda d_dA, dA_dS: softmax_d_over_dSignals_from_d_over_dActivations(d_dA, dA_dS)}
    return funcs___dict[nameOfActivationFunc](d_over_dActivations, dActivations_over_dSignals)