from legoML._common import func_onDict, dictOfFuncs, renameKeys_funcOnDict, varName_fromVarTuple


class Piece:
    def __init__(self, *args, **kwargs):
        inKeys = set()
        func = lambda: None
        outKeys = set()
        if len(args) > 0:
            funcs___dict = {}
            for eachList in args:
                argKeysFromInKeys_inThisList___dict, func_inThisList, outKey_inThisList = eachList
                inKeys.update(argKeysFromInKeys_inThisList___dict.values())
                funcs___dict[outKey_inThisList] = func_onDict(func_inThisList, argKeysFromInKeys_inThisList___dict)
                outKeys.add(outKey_inThisList)
            func = lambda ins___dict: dictOfFuncs(funcs___dict, ins___dict)
        self.inKeys = inKeys
        self.func = func
        self.outKeys = outKeys

    def installPiece(self, toNewInKeys_fromOldInKeys___dict, toNewOutKeys_fromOldOutKeys___dict):
        inKeys = self.inKeys.copy()
        for newKey in toNewInKeys_fromOldInKeys___dict.keys():
            inKeys.remove(toNewInKeys_fromOldInKeys___dict[newKey])
            inKeys.add(newKey)
        outKeys = self.outKeys.copy()
        for newKey in toNewOutKeys_fromOldOutKeys___dict.keys():
            outKeys.remove(toNewOutKeys_fromOldOutKeys___dict[newKey])
            outKeys.add(newKey)
        p = Piece()
        p.inKeys = inKeys
        p.func = lambda ins___dict: renameKeys_funcOnDict(self.func, toNewInKeys_fromOldInKeys___dict,
                                                          toNewOutKeys_fromOldOutKeys___dict, ins___dict)
        p.outKeys = outKeys
        return p

    def runPiece(self, dictObj):
        d = dictObj.copy()
        ins___dict = {}
        for varTuple in self.inKeys:
            if isinstance(varTuple, tuple):
                varName, varIndexOrKey = varTuple
                ins___dict[varTuple] = d[varName][varIndexOrKey]
            else:
                ins___dict[varTuple] = d[varTuple]
        outs___dict = self.func(ins___dict)
        for varTuple in self.outKeys:
            if isinstance(varTuple, tuple):
                varName, varIndexOrKey = varTuple
                d[varName] = d[varName].copy()
                d[varName][varIndexOrKey] = outs___dict[varTuple]
            else:
                d[varTuple] = outs___dict[varTuple]
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