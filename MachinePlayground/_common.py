def renameKeys(dictObj, toNewKeys_fromOldKeys___dict):
    d = dictObj.copy()
    for newKey in toNewKeys_fromOldKeys___dict.keys():
        oldKey = toNewKeys_fromOldKeys___dict[newKey]
        v = d[oldKey]
        del d[oldKey]
        d[newKey] = v
    return d


def subDict_withNewKeys(dictObj, toNewKeys_fromOldKeys___dict):
    # extract values from dictionary and assign them new keys
    d = {}
    for newKey in toNewKeys_fromOldKeys___dict.keys():
        d[newKey] = dictObj[toNewKeys_fromOldKeys___dict[newKey]]
    return d


def swapKeysValues(dictObj):
    d = {}
    for k in dictObj.keys():
        d[dictObj[k]] = k
    return d


def func_onDict(func, toArgKeys_fromDictKeys___dict):
    # return a lambda that:
    # (1) extracts values from a dictionary and assign them new keys in a new dictionary
    # (2) pass the new dictionary into a function
    return lambda d: func(**subDict_withNewKeys(d, toArgKeys_fromDictKeys___dict))


def dictOfFuncs(funcs___dict, x):
    # computes a dictionary of values, given:
    # (1) a dictionary of functions, each of which can take in:
    # (2) a certain same input X
    d = {}
    for k in funcs___dict.keys():
        d[k] = funcs___dict[k](x)
    return d


def renameKeys_funcOnDict(func, toNewInKeys_fromOldInKeys___dict, toNewOutKeys_fromOldOutKeys___dict, ins___dict):
    toOldInKeys_fromNewInKeys___dict = swapKeysValues(toNewInKeys_fromOldInKeys___dict)
    return renameKeys(func(renameKeys(ins___dict, toOldInKeys_fromNewInKeys___dict)),
                      toNewOutKeys_fromOldOutKeys___dict)


def varName_fromVarTuple(varTuple):
    # return variable name from tuple consisting of variable name and index/key
    if isinstance(varTuple, tuple):
        return varTuple[0]
    else:
        return varTuple


def updateDictValues(dict_toUpdate, toKeysOfDictToUpdate_fromKeysOfDictWithValues____dict, dict_withValues):
    # transfer values from DICT_WITHVALUES to DICT_TOUPDATE, according to the relationships defined in the dict
    # containing the mappings between variable names in those two dicts. Variables can be indexed by a number
    # (for list/tuple) or key (for dict)
    d = dict_toUpdate.copy()
    for varTuple_ofDictToUpdate in toKeysOfDictToUpdate_fromKeysOfDictWithValues____dict.keys():
        varTuple_ofDictWithValues = toKeysOfDictToUpdate_fromKeysOfDictWithValues____dict[varTuple_ofDictToUpdate]
        if isinstance(varTuple_ofDictWithValues, tuple):
            varName, varIndexOrKey = varTuple_ofDictWithValues
            value = dict_withValues[varName][varIndexOrKey]
        else:
            value = dict_withValues[varTuple_ofDictWithValues]
        if isinstance(varTuple_ofDictToUpdate, tuple):
            varName, varIndexOrKey = varTuple_ofDictToUpdate
            d[varName] = d[varName].copy()
            d[varName][varIndexOrKey] = value
        else:
            d[varTuple_ofDictToUpdate] = value
    return d


def approxGrad(func, arrayA, epsilon = 1e-6):
    g = zeros(arrayA.shape)
    for i in range(arrayA.size):
        a_plus = arrayA.copy()
        a_plus.flat[i] += epsilon
        a_minus = arrayA.copy()
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