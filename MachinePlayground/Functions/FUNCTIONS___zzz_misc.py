from numpy import zeros
from copy import deepcopy


def as_strings(iterable_object):
    strings = []
    for i in iterable_object:
        strings += [str(i)]
    return strings


def sympy_args(sympy_expression):
    args = set()
    try:
        args = set(as_strings(sympy_expression.args))
    except:
        pass
    return args


def sympy_subs(sympy_expression, args_and_values___dict={}, **kw_args_and_values___dict):
    if args_and_values___dict:
        args_and_values___dict = deepcopy(args_and_values___dict)   # just to be careful #
        args_and_values___dict.update(kw_args_and_values___dict)
    else:
        args_and_values___dict = kw_args_and_values___dict
    try:
        sympy_expression = sympy_expression.subs(args_and_values___dict)
    except:
        pass
    return sympy_expression


#def subDict_withNewKeys(dictObj, toNewKeys_fromOldKeys___dict):
#    # extract values from dictionary and assign them new keys
#    return {newKey: dictObj[toNewKeys_fromOldKeys___dict[newKey]] for newKey in toNewKeys_fromOldKeys___dict}


#def swapKeysValues(dictObj):
#    return {dictObj[k]: k for k in dictObj}


#def func_onDict(func, toArgKeys_fromDictKeys___dict):
    # return a lambda that:
    # (1) extracts values from a dictionary and assign them new keys in a new dictionary
    # (2) pass the new dictionary into a function
#    return lambda d: func(**subDict_withNewKeys(d, toArgKeys_fromDictKeys___dict))


#def dictOfFuncs(funcs___dict, x):
    # computes a dictionary of values, given:
    # (1) a dictionary of functions, each of which can take in:
    # (2) a certain same input X
#    return {k: funcs___dict[k](x) for k in funcs___dict}



#def updateDictValues(dict_toUpdate, toKeysOfDictToUpdate_fromKeysOfDictWithValues____dict, dict_withValues):
    # transfer values from DICT_WITHVALUES to DICT_TOUPDATE, according to the relationships defined in the dict
    # containing the mappings between variable names in those two dicts. Variables can be indexed by a number
    # (for list/tuple) or key (for dict)
#    d = dict_toUpdate.copy()
#    for varTuple_ofDictToUpdate in toKeysOfDictToUpdate_fromKeysOfDictWithValues____dict.keys():
#        varTuple_ofDictWithValues = toKeysOfDictToUpdate_fromKeysOfDictWithValues____dict[varTuple_ofDictToUpdate]
#        if isinstance(varTuple_ofDictWithValues, tuple):
#            varName, varIndexOrKey = varTuple_ofDictWithValues
#            value = dict_withValues[varName][varIndexOrKey]
#        else:
#            value = dict_withValues[varTuple_ofDictWithValues]
#        if isinstance(varTuple_ofDictToUpdate, tuple):
#            varName, varIndexOrKey = varTuple_ofDictToUpdate
#            d[varName] = d[varName].copy()
#            d[varName][varIndexOrKey] = value
#        else:
#            d[varTuple_ofDictToUpdate] = value
#    return d


def approx_gradients(func, array_a, epsilon = 1e-6):
    g = zeros(array_a.shape)
    for i in range(array_a.size):
        a_plus = array_a.copy()
        a_plus.flat[i] += epsilon
        a_minus = array_a.copy()
        a_minus.flat[i] -= epsilon
        g.flat[i] = (func(a_plus) - func(a_minus)) / (2 * epsilon)
    return g



# ---------



def asList(x):
    # convert any non-list to list
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def asTuple(x):
    # convert any non-tuple to tuple
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def setVar(varTuple, value):
    if isinstance(varTuple, tuple):
        varName, varIndexOrKey = varTuple
        globals()[varName][varIndexOrKey] = value
    else:
        globals()[varTuple] = value


def getVar(varTuple):
    if isinstance(varTuple, tuple):
        varName, varIndexOrKey = varTuple
        return globals()[varName][varIndexOrKey]
    else:
        return globals()[varTuple]