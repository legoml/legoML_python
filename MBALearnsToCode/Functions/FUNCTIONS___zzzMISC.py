from copy import deepcopy
from numpy import zeros
from sympy.matrices import MatrixSymbol
from frozen_dict import FrozenDict
from MBALearnsToCode.Functions.FUNCTIONS___SymPy import is_non_atomic_sympy_expression, sympy_allclose


def combine_dict_and_kwargs(dict_object, kwargs):
    d = kwargs
    if dict_object:
        d.update(dict_object)
    return d


def rename_dict_keys(dict_object, to_new_keys_from_old_keys___dict):
    d = deepcopy(dict_object)   # just to be careful #
    for new_key, old_key in to_new_keys_from_old_keys___dict.items():
        value = d[old_key]
        del d[old_key]
        d[new_key] = value
    return d


def merge_dicts(dict_0, dict_1):
    d = deepcopy(dict_0)
    for key, value in dict_1.items():
        if (key not in dict_0) or (value is not None):
            d[key] = value
    return d


def dicts_all_close(*dicts, **kwargs):
    if len(dicts) == 2:
        if set(dicts[0]) == set(dicts[1]):
            for key in dicts[0]:
                if not sympy_allclose(dicts[0][key], dicts[1][key], **kwargs):
                    return False
            return True
        else:
            return False
    else:
        for i in range(1, len(dicts)):
            if not dicts_all_close(dicts[0], dicts[i]):
                return False
        return True


def approx_gradients(function, argument_array, epsilon=1e-6):
    g = zeros(argument_array.shape)
    for i in range(argument_array.size):
        a_plus = argument_array.copy()
        a_plus.flat[i] += epsilon
        a_minus = argument_array.copy()
        a_minus.flat[i] -= epsilon
        g.flat[i] = (function(a_plus) - function(a_minus)) / (2 * epsilon)
    return g


def shift_time_subscripts(obj, t, *matrix_symbols_to_shift):
    if isinstance(obj, FrozenDict):
        return FrozenDict({shift_time_subscripts(key, t): shift_time_subscripts(value, t)
                           for key, value in obj.items()})
    elif isinstance(obj, tuple):
        if len(obj) == 2 and not(isinstance(obj[0], (int, float))) and isinstance(obj[1], int):
            return shift_time_subscripts(obj[0], t), obj[1] + t
        else:
            return tuple(shift_time_subscripts(item, t) for item in obj)
    elif isinstance(obj, list):
        return [shift_time_subscripts(item, t) for item in obj]
    elif isinstance(obj, set):
        return {shift_time_subscripts(item, t) for item in obj}
    elif isinstance(obj, dict):
        return {shift_time_subscripts(key, t): shift_time_subscripts(value, t) for key, value in obj.items()}
    elif isinstance(obj, MatrixSymbol):
        args = obj.args
        if isinstance(args[0], tuple):
            return MatrixSymbol(shift_time_subscripts(args[0], t), args[1], args[2])
        else:
            return obj
    elif is_non_atomic_sympy_expression(obj):
        return obj.xreplace({matrix_symbol: shift_time_subscripts(matrix_symbol, t)
                             for matrix_symbol in matrix_symbols_to_shift})
    else:
        return obj