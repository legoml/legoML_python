from copy import deepcopy
from MachinePlayground._common import varName_fromVarTuple


class Piece:
    def __init__(self, forwards = {}, backwards = {}, *args, **kwargs):
        self.inKeys = set()
        self.outKeys = set(forwards.keys())
        self.forwards = deepcopy(forwards)
        self.forwardsToFrom = {}
        self.backwards = deepcopy(backwards)
        self.backwardsToFrom = {}
        if len(forwards) > 0:
            for outKey, func_andToFrom in forwards.items():
                inKeys_forThisOutKey = func_andToFrom[1].values()
                self.inKeys.update(inKeys_forThisOutKey)
                self.forwardsToFrom[outKey] = list(inKeys_forThisOutKey)
            if len(backwards) > 0:
                for inKey, func_andToFrom in backwards.items():
                    self.backwardsToFrom[inKey] = list(func_andToFrom[1].values())


    def copyPiece(self, fromOldKeys_toNewKeys___dict):

        forwards = {}
        for outKey, func_andToFrom in self.forwards.items():
            if outKey in fromOldKeys_toNewKeys___dict:
                newOutKey = fromOldKeys_toNewKeys___dict[outKey]
            else:
                newOutKey = outKey
            newFunc_andToFrom = deepcopy(func_andToFrom)
            for argKey, inKey in func_andToFrom[1].items():
                if inKey in fromOldKeys_toNewKeys___dict:
                    newFunc_andToFrom[1][argKey] = fromOldKeys_toNewKeys___dict[inKey]
            forwards[newOutKey] = newFunc_andToFrom

        backwards = {}
        for inKey, func_andToFrom in self.backwards.items():
            if inKey in fromOldKeys_toNewKeys___dict:
                newInKey = fromOldKeys_toNewKeys___dict[inKey]
            elif (inKey[0] == 'dOverD') and (inKey[1] in fromOldKeys_toNewKeys___dict):
                newInKey = ('dOverD', fromOldKeys_toNewKeys___dict[inKey[1]])
            else:
                newInKey = inKey
            newFunc_andToFrom = deepcopy(func_andToFrom)
            for argKey, inOutKey in func_andToFrom[1].items():
                if inOutKey in fromOldKeys_toNewKeys___dict:
                    newFunc_andToFrom[1][argKey] = fromOldKeys_toNewKeys___dict[inOutKey]
                elif (inOutKey[0] == 'dOverD') and (inOutKey[1] in fromOldKeys_toNewKeys___dict):
                    newFunc_andToFrom[1][argKey] = ('dOverD', fromOldKeys_toNewKeys___dict[inOutKey[1]])
            backwards[newInKey] = newFunc_andToFrom

        return Piece(forwards, backwards)


    def runPiece(self, dictObj, forwardOutKeys = set(), dKey_and_backwardInKeys = None):

        d = dictObj.copy()

        if forwardOutKeys is not None:
            if len(forwardOutKeys) > 0:
                outKeys = forwardOutKeys
            else:
                outKeys = self.outKeys
            for outVarTuple in outKeys:
                func, arguments = deepcopy(self.forwards[outVarTuple])
                for argKey, inVarTuple in arguments.items():
                    if isinstance(inVarTuple, tuple):
                        inVarName, inVarIndexOrKey = inVarTuple
                        arguments[argKey] = d[inVarName][inVarIndexOrKey]
                    else:
                        arguments[argKey] = d[inVarTuple]
                value = func(arguments)
                if isinstance(outVarTuple, tuple):
                    outVarName, outVarIndexOrKey = outVarTuple
                    d[outVarName] = d[outVarName].copy()
                    d[outVarName][outVarIndexOrKey] = value
                else:
                    d[outVarTuple] = value

        if dKey_and_backwardInKeys is not None:
            dKey, backwardInKeys = dKey_and_backwardInKeys
            if len(backwardInKeys) > 0:
                backwardInKeys = set(map(lambda k: ('dOverD', k), backwardInKeys))
                backwards = {inKey: self.backwards[inKey] for inKey in backwardInKeys}
            else:
                backwards = self.backwards
            for backwardVarTuple, func_andToFrom in backwards.items():
                func, arguments = deepcopy(func_andToFrom)
                for argKey, varTuple in arguments.items():
                    if varTuple[0] == 'dOverD':
                        varTuple_forDifferentiation = varTuple[1]
                        if isinstance(varTuple_forDifferentiation, tuple):
                            varName, varIndexOrKey = varTuple_forDifferentiation
                            arguments[argKey] = d[('dOverD', dKey, varName)][varIndexOrKey]
                        else:
                            arguments[argKey] = d[('dOverD', dKey, varTuple_forDifferentiation)]
                    elif isinstance(varTuple, tuple):
                        varName, varIndexOrKey = varTuple
                        arguments[argKey] = d[varName][varIndexOrKey]
                    else:
                        arguments[argKey] = d[varTuple]
                value = func(arguments)
                if backwardVarTuple[0] == 'dOverD':
                    varTuple_forDifferentiation = backwardVarTuple[1]
                    if isinstance(varTuple_forDifferentiation, tuple):
                        varName, varIndexOrKey = varTuple_forDifferentiation
                        t = ('dOverD', dKey, varName)
                        d[t] = d[t].copy()
                        d[t][varIndexOrKey] = value
                    else:
                        d[('dOverD', dKey, varTuple_forDifferentiation)] = value
                elif isinstance(backwardVarTuple, tuple):
                    varName, varIndexOrKey = backwardVarTuple
                    d[varName] = d[varName].copy()
                    d[varName][varIndexOrKey] = value
                else:
                    d[backwardVarTuple] = value

        return d





class Operation:
    def __init__(self, *args, **kwargs):
        objects = set()
        steps = []
        for piece in args:
            for k in piece.inKeys:
                objects.add(varName_fromVarTuple(k))
            for k in piece.outKeys:
                objects.add(varName_fromVarTuple(k))
            steps += [piece]
        self.objects = objects
        self.steps = steps

    def runOperation(self, dictObj):
        d = dictObj.copy()
        for p in self.steps:
            d = p.runPiece(d)
        return d


class Group:
    def __init__(self, groupKeysFromObjectNames___dict, processes, *args, **kwargs):
        self.groupKeysFromObjectNames = groupKeysFromObjectNames___dict
        self.processes = processes

    def installGroup(self, toNewInKeys_fromOldInKeys___dict):
        d = 0
        return d