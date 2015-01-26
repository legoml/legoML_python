from numpy import allclose, ones, ndarray
from copy import deepcopy
from MachinePlayground._common import approxGradients, varName_fromVarTuple
import itertools



def isOverD(varTuple):
    return (varTuple[0] == 'dOverD') and (len(varTuple) == 2)


def isDOverD(varTuple):
    return (varTuple[0] == 'dOverD') and (len(varTuple) == 3)


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
                inKeys_forThisOutKey = set(func_andToFrom[1].values())
                self.inKeys.update(inKeys_forThisOutKey)
                self.forwardsToFrom[outKey] = inKeys_forThisOutKey
        if len(backwards) > 0:
           for inKey, func_andToFrom in backwards.items():
               self.backwardsToFrom[inKey] = set(func_andToFrom[1].values())


    def install(self, fromOldKeys_toNewKeys___dict):

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
        for backwardInKey, func_andToFrom in self.backwards.items():
            if isOverD(backwardInKey) and (backwardInKey[1] in fromOldKeys_toNewKeys___dict):
                newBackwardInKey = ('dOverD', fromOldKeys_toNewKeys___dict[backwardInKey[1]])
            elif isDOverD(backwardInKey):
                if backwardInKey[1] in fromOldKeys_toNewKeys___dict:
                    dKey = fromOldKeys_toNewKeys___dict[backwardInKey[1]]
                else:
                    dKey = backwardInKey[1]
                if backwardInKey[2] in fromOldKeys_toNewKeys___dict:
                    overDKey = fromOldKeys_toNewKeys___dict[backwardInKey[2]]
                else:
                    overDKey = backwardInKey[2]
                newBackwardInKey = ('dOverD', dKey, overDKey)
            else:
                newBackwardInKey = backwardInKey
            newFunc_andToFrom = deepcopy(func_andToFrom)
            for argKey, backwardInOutKey in func_andToFrom[1].items():
                if backwardInOutKey in fromOldKeys_toNewKeys___dict:
                    newFunc_andToFrom[1][argKey] = fromOldKeys_toNewKeys___dict[backwardInOutKey]
                elif isOverD(backwardInOutKey) and (backwardInOutKey[1] in fromOldKeys_toNewKeys___dict):
                    newFunc_andToFrom[1][argKey] = ('dOverD',
                                                    fromOldKeys_toNewKeys___dict[backwardInOutKey[1]])
                elif isDOverD(backwardInOutKey):
                    if backwardInOutKey[1] in fromOldKeys_toNewKeys___dict:
                        dKey = fromOldKeys_toNewKeys___dict[backwardInOutKey[1]]
                    else:
                        dKey = backwardInOutKey[1]
                    if backwardInOutKey[2] in fromOldKeys_toNewKeys___dict:
                        overDKey = fromOldKeys_toNewKeys___dict[backwardInOutKey[2]]
                    else:
                        overDKey = backwardInOutKey[2]
                    newFunc_andToFrom[1][argKey] = ('dOverD', dKey, overDKey)
            backwards[newBackwardInKey] = newFunc_andToFrom

        return Piece(forwards, backwards)


    def run(self, dictObj, forwardOutKeys = set(), dKey_and_backwardInKeys = None):

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
                        inVarName, inVarKey = inVarTuple
                        arguments[argKey] = d[inVarName][inVarKey]
                    else:
                        arguments[argKey] = d[inVarTuple]
                value = func(**arguments)
                if isinstance(outVarTuple, tuple):
                    outVarName, outVarKey = outVarTuple
                    if outVarName in d:
                        d[outVarName] = d[outVarName].copy()
                        d[outVarName][outVarKey] = value
                    else:
                        d[outVarName] = {outVarKey: value}
                else:
                    d[outVarTuple] = value

        if dKey_and_backwardInKeys is not None:
            dKey, backwardInKeys = dKey_and_backwardInKeys
            if len(backwardInKeys) > 0:
                list1 = list(map(lambda backwardInKey: ('dOverD', backwardInKey), backwardInKeys))
                list2 = list(map(lambda backwardInKey: ('dOverD', dKey, backwardInKey), backwardInKeys))
                backwardInKeys = set(list1 + list2)
                backwards = {backwardInKey: self.backwards[backwardInKey]
                             for backwardInKey in backwardInKeys.intersection(self.backwards)}
            else:
                backwards = self.backwards
            for backwardVarTuple, func_andToFrom in backwards.items():
                func, arguments = deepcopy(func_andToFrom)
                for argKey, varTuple in arguments.items():
                    if isOverD(varTuple):
                        overD = varTuple[1]
                        if isinstance(overD, tuple):
                            varName, varKey = overD
                            arguments[argKey] = d[('dOverD', dKey, varName)][varKey]
                        else:
                            arguments[argKey] = d[('dOverD', dKey, overD)]
                    elif isDOverD(varTuple):
                        overD = varTuple[2]
                        if isinstance(overD, tuple):
                            varName, varKey = overD
                            arguments[argKey] = d[('dOverD', dKey, varName)][varKey]
                        else:
                            arguments[argKey] = d[('dOverD', dKey, overD)]
                    elif isinstance(varTuple, tuple):
                        varName, varKey = varTuple
                        arguments[argKey] = d[varName][varKey]
                    else:
                        arguments[argKey] = d[varTuple]
                value = func(**arguments)
                if isOverD(backwardVarTuple):
                    overD = backwardVarTuple[1]
                if isDOverD(backwardVarTuple):
                    overD = backwardVarTuple[2]
                if isinstance(overD, tuple):
                    varName, varKey = overD
                    t = ('dOverD', dKey, varName)
                    if t in d:
                        d[t] = d[t].copy()
                        d[t][varKey] = value
                    else:
                        d[t] = {varKey: value}
                else:
                    d[('dOverD', dKey, overD)] = value

        return d


    def checkGradients(self, ins___dict):

        def sumOut(in___dict, outKey):
            d = ins___dict.copy()
            for inKey, inValue in in___dict.items():
                d[inKey] = inValue
            out = self.run(d)[outKey]
            if isinstance(out, ndarray):
                return out.sum()
            else:
                return out

        vars = self.run(ins___dict)
        dKeys = set()
        dSumKeys = set()
        for outKey in self.outKeys:
            if isinstance(vars[outKey], float):
                for varTuple in self.backwards:
                    if isDOverD(varTuple) and (varTuple[1] == outKey):
                        dKeys.add(outKey)
            elif isinstance(vars[outKey], ndarray):
                vars[('dOverD', 'SUM___' + outKey, outKey)] = ones(vars[outKey].shape)
                dSumKeys.add(outKey)

        backwardInKeys = set()
        for backwardVarTuple in self.backwards:
            if isOverD(backwardVarTuple):
                backwardInKeys.add(backwardVarTuple[1])
            elif isDOverD(backwardVarTuple):
                backwardInKeys.add(backwardVarTuple[2])

        for outKey in dKeys:
            vars = self.run(vars, forwardOutKeys = None, dKey_and_backwardInKeys = (outKey, backwardInKeys))
        for outKey in dSumKeys:
            vars = self.run(vars, forwardOutKeys = None, dKey_and_backwardInKeys = ('SUM___' + outKey, backwardInKeys))

        check = True
        for inKey, outKey in itertools.product(backwardInKeys, dKeys):
            approxGrad = approxGradients(lambda v: sumOut({inKey: v}, outKey), ins___dict[inKey])
            check = check and allclose(approxGrad, vars[('dOverD', outKey, inKey)], rtol = 1.e-3, atol = 1.e-6)
        for inKey, outKey in itertools.product(backwardInKeys, dSumKeys):
            approxGrad = approxGradients(lambda v: sumOut({inKey: v}, outKey), ins___dict[inKey])
            check = check and allclose(approxGrad, vars[('dOverD', 'SUM___' + outKey, inKey)],
                                       rtol = 1.e-3, atol = 1.e-6)

        return check



def step_withDefaultForwardsAndBackwards(step):
    if isinstance(step, list):
        if len(step) == 1:
            return step + [set(), None]
        elif len(step) == 2:
            return step + [None]
        else:
            return step
    else:
        return [step, set(), None]


class Process:
    def __init__(self, *args, **kwargs):
        vars = set()
        steps = []
        for step in args:
            s = step_withDefaultForwardsAndBackwards(step)
            piece = s[0]
            for inVarTuple in piece.inKeys:
                vars.add(varName_fromVarTuple(inVarTuple))
            for outVarTuple in piece.outKeys:
                vars.add(varName_fromVarTuple(outVarTuple))
            steps += s
        self.vars = vars
        self.steps = steps

    def addSteps(self, *args):
        for step in args:
            s = step_withDefaultForwardsAndBackwards(step)
            piece = s[0]
            for inVarTuple in piece.inKeys:
                self.vars.add(varName_fromVarTuple(inVarTuple))
            for outVarTuple in piece.outKeys:
                self.vars.add(varName_fromVarTuple(outVarTuple))
            self.steps += s

    def run(self, dictObj):
        d = dictObj.copy()
        for step in self.steps:
            piece, forwardOutKeys, dKey_and_backwardInKeys = step
            d = piece.run(forwardOutKeys, dKey_and_backwardInKeys)
        return d


def connectProcesses(*args, **kwargs):
    p = deepcopy(args[0])
    for process in args[1:]:
        p.vars.update(process.vars)
        p.steps += process.steps
    return p


class Program:
    def __init__(self, vars___dict, pieces___dict, processes___dict, *args, **kwargs):
        self.vars = vars___dict
        self.pieces = pieces___dict
        self.processes = processes___dict

    def install(self, toNewKeys_fromOldKeys___dict):
        d = 0

        return d

    def run(self, piece_or_process, *args, *kwargs):
        self.vars = piece_or_process.run(self.vars, *args, **kwargs)



class Project:
    def __init__(self, *args, **kwargs):
        self.vars = dict.fromkeys(args)

    def setVar(self, *args):
        pass

    def getVar(self, *args):
        pass