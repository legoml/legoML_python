def itself(x):
    # just return argument object itself
    return x



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



def subDict_withNewKeys(toNewKeys_fromOldKeys___dict, dictObj):
    # extract values from dictionary and assign them new keys
    d = {}
    for k in toNewKeys_fromOldKeys___dict.keys():
        d[k] = dictObj[toNewKeys_fromOldKeys___dict[k]]
    return d



def func_onDict(func, argKeysFromDictKeys___dict):
    # return a lambda that:
    # (1) extracts values from a dictionary and assign them new keys in a new dictionary
    # (2) pass the new dictionary into a function
    func_subDict = lambda d: subDict_withNewKeys(argKeysFromDictKeys___dict, d)
    return lambda d: func(**func_subDict(d))



def dictOfFuncs(funcs___dict, x):
    # computes a dictionary of values, given:
    # (1) a dictionary of functions, each of which can take in:
    # (2) a certain same input X
    d = {}
    for k in funcs___dict.keys():
        d[k] = funcs___dict[k](x)
    return d