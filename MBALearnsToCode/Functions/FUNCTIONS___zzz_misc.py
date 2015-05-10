from numpy import zeros
from copy import deepcopy
from sympy.matrices import MatrixSymbol
from frozen_dict import FrozenDict
from MBALearnsToCode.Functions.FUNCTIONS___sympy import is_non_atomic_sympy_expression


def as_tuple(x):
    # convert any non-tuple to tuple
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def as_list(x):
    # convert any non-list to list
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def as_strings(obj):
    if isinstance(obj, str):
        return str
    elif hasattr(obj, '__iter__'):
        strings = []
        for i in obj:
            strings += [str(i)]
        return strings
    else:
        return str(obj)


def combine_dict_and_kwargs(dict_object, kwargs):
    if dict_object:
        d = dict_object.copy()
        d.update(deepcopy(kwargs))
    else:
        d = deepcopy(kwargs)
    return d


def dict_with_string_keys(dict_obj):
    d = {}
    for key, value in dict_obj.items():
        d[str(key)] = value
    return d


def frozen_dict_with_string_keys(frozen_dict):
    fdict = {}
    for key, value in frozen_dict.items():
        fdict[str(key)] = value
    return FrozenDict(fdict)


def approx_gradients(func, array_a, epsilon = 1e-6):
    g = zeros(array_a.shape)
    for i in range(array_a.size):
        a_plus = array_a.copy()
        a_plus.flat[i] += epsilon
        a_minus = array_a.copy()
        a_minus.flat[i] -= epsilon
        g.flat[i] = (func(a_plus) - func(a_minus)) / (2 * epsilon)
    return g


def sympy_string_args(sympy_expression):
    args = set()
    try:
        args = set(as_strings(sympy_expression.args))
    except:
        pass
    return args


def sympy_subs(sympy_expression, args_and_values___dict={}, **kw_args_and_values___dict):
    args_and_values___dict = combine_dict_and_kwargs(args_and_values___dict, kw_args_and_values___dict)
    try:
        sympy_expression = sympy_expression.subs(args_and_values___dict)
    except:
        pass
    return sympy_expression


def rename_dict_keys(dict_object, to_new_keys_from_old_keys___dict):
    d = deepcopy(dict_object)   # just to be careful #
    for new_key, old_key in to_new_keys_from_old_keys___dict.items():
        value = d[old_key]
        del d[old_key]
        d[new_key] = value
    return d


def merge_dicts(dict_1, dict_2):
    d = deepcopy(dict_1)
    for key, value in dict_2.items():
        if (key not in dict_1) or (value is not None):
            d[key] = value
    return d


def shift_time_subscripts(obj, t, *matrix_symbols_to_shift):
    if isinstance(obj, FrozenDict):
        return FrozenDict({shift_time_subscripts(key, t): shift_time_subscripts(value, t)
                           for key, value in obj.items()})
    elif isinstance(obj, tuple):
        if len(obj) == 2 and isinstance(obj[1], int):
            tup = (shift_time_subscripts(obj[0], t), obj[1] + t)
            return tup
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