from numpy import zeros
from copy import deepcopy
from frozen_dict import FrozenDict


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