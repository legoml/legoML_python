from copy import copy, deepcopy
from sympy import Atom
from frozen_dict import FrozenDict


def is_non_atomic_sympy_expression(obj):
    return hasattr(obj, 'doit') and not isinstance(obj, Atom)


def sympy_xreplace(obj, xreplace_dict={}):
    if isinstance(obj, tuple):
        return tuple(sympy_xreplace(item, xreplace_dict) for item in obj)
    elif isinstance(obj, list):
        return [sympy_xreplace(item, xreplace_dict) for item in obj]
    elif isinstance(obj, set):
        return set(sympy_xreplace(item, xreplace_dict) for item in obj)
    elif isinstance(obj, frozenset):
        return frozenset(sympy_xreplace(item, xreplace_dict) for item in obj)
    elif isinstance(obj, dict):
        return {sympy_xreplace(key, xreplace_dict): sympy_xreplace(value, xreplace_dict)
                for key, value in obj.items()}
    elif isinstance(obj, FrozenDict):
        return FrozenDict({sympy_xreplace(key, xreplace_dict): sympy_xreplace(value, xreplace_dict)
                           for key, value in obj.items()})
    elif hasattr(obj, 'xreplace'):
        return obj.xreplace(xreplace_dict)
    else:
        return deepcopy(obj)


def sympy_xreplace_doit_explicit(obj, xreplace_dict={}):
    obj = copy(obj)
    if isinstance(obj, dict):
        obj = {key: sympy_xreplace_doit_explicit(value, xreplace_dict) for key, value in obj.items()}
    else:
        # xreplace into all nodes of the expression tree first
        if xreplace_dict:
            obj = obj.xreplace(xreplace_dict)
        # traverse the tree to compute
        if is_non_atomic_sympy_expression(obj):
            args = []
            for arg in obj.args:
                # compute each argument
                args += [sympy_xreplace_doit_explicit(arg)]
            # reconstruct function
            obj = obj.func(*args)
            # try to do it if expression is complete
            try:
                obj = obj.doit()
            except:
                pass
            # try to make it explicit if possible
            try:
                obj = obj.as_explicit()
            except:
                pass
    return obj


def sympy_xreplace_doit_explicit_evalf(obj, xreplace_dict={}):
    obj = sympy_xreplace_doit_explicit(obj, xreplace_dict)
    # try evaluating out to get numerical value
    if isinstance(obj, dict):
        for key, value in obj.items():
            try:
                obj[key] = value.evalf()
            except:
                pass
    else:
        try:
            obj = obj.evalf()
        except:
            pass
    return obj