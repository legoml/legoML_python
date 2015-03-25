from copy import deepcopy
import itertools
from numpy import allclose, ones, ndarray
from MachinePlayground.Functions.FUNCTIONS___zzz_misc import approx_gradients



class Piece:
    def __init__(self, forwards = {}, backwards = {}):
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


    def install(self, change_keys = {}, change_vars = {}):

        piece = deepcopy(self)

        if change_keys:

            forwards = {}
            for outKey, func_andToFrom in piece.forwards.items():
                if outKey in change_keys:
                    newOutKey = change_keys[outKey]
                else:
                    newOutKey = outKey
                newFunc_andToFrom = deepcopy(func_andToFrom)
                for argKey, inKey in func_andToFrom[1].items():
                    if inKey in change_keys:
                        newFunc_andToFrom[1][argKey] = change_keys[inKey]
                forwards[newOutKey] = newFunc_andToFrom

            backwards = {}
            for backwardInKey, func_andToFrom in piece.backwards.items():
                if isOverD(backwardInKey) and (backwardInKey[1] in change_keys):
                    newBackwardInKey = ('DOVERD', change_keys[backwardInKey[1]])
                elif isDOverD(backwardInKey):
                    if backwardInKey[1] in change_keys:
                       dKey = change_keys[backwardInKey[1]]
                    else:
                        dKey = backwardInKey[1]
                    if backwardInKey[2] in change_keys:
                        overDKey = change_keys[backwardInKey[2]]
                    else:
                        overDKey = backwardInKey[2]
                    newBackwardInKey = ('DOVERD', dKey, overDKey)
                else:
                    newBackwardInKey = backwardInKey
                newFunc_andToFrom = deepcopy(func_andToFrom)
                for argKey, backwardInOutKey in func_andToFrom[1].items():
                    if backwardInOutKey in change_keys:
                        newFunc_andToFrom[1][argKey] = change_keys[backwardInOutKey]
                    elif isOverD(backwardInOutKey) and (backwardInOutKey[1] in change_keys):
                        newFunc_andToFrom[1][argKey] = ('DOVERD',
                                                        change_keys[backwardInOutKey[1]])
                    elif isDOverD(backwardInOutKey):
                        if backwardInOutKey[1] in change_keys:
                            dKey = change_keys[backwardInOutKey[1]]
                        else:
                            dKey = backwardInOutKey[1]
                        if backwardInOutKey[2] in change_keys:
                            overDKey = change_keys[backwardInOutKey[2]]
                        else:
                            overDKey = backwardInOutKey[2]
                        newFunc_andToFrom[1][argKey] = ('DOVERD', dKey, overDKey)
                backwards[newBackwardInKey] = newFunc_andToFrom

            piece = Piece(forwards, backwards)

        if change_vars:
            change_keys___dict = {}
            for key in (piece.inKeys).union(piece.outKeys):
                change_keys___dict[key] = change_var_in_piece_key(key, change_vars)
            piece = piece.install(change_keys = change_keys___dict)

        return piece



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
                        if inVarName in d:
                            arguments[argKey] = d[inVarName][inVarKey]
                    else:
                        if inVarTuple in d:
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
                list1 = list(map(lambda backwardInKey: ('DOVERD', backwardInKey), backwardInKeys))
                list2 = list(map(lambda backwardInKey: ('DOVERD', dKey, backwardInKey), backwardInKeys))
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
                            if varName in d:
                                arguments[argKey] = d[('DOVERD', dKey, varName)][varKey]
                        else:
                            if overD in d:
                                arguments[argKey] = d[('DOVERD', dKey, overD)]
                    elif isDOverD(varTuple):
                        overD = varTuple[2]
                        if isinstance(overD, tuple):
                            varName, varKey = overD
                            if varName in d:
                                arguments[argKey] = d[('DOVERD', dKey, varName)][varKey]
                        else:
                            if overD in d:
                                arguments[argKey] = d[('DOVERD', dKey, overD)]
                    elif isinstance(varTuple, tuple):
                        varName, varKey = varTuple
                        if varName in d:
                            arguments[argKey] = d[varName][varKey]
                    else:
                        if varTuple in d:
                            arguments[argKey] = d[varTuple]
                value = func(**arguments)
                if isOverD(backwardVarTuple):
                    overD = backwardVarTuple[1]
                if isDOverD(backwardVarTuple):
                    overD = backwardVarTuple[2]
                if isinstance(overD, tuple):
                    varName, varKey = overD
                    t = ('DOVERD', dKey, varName)
                    if t in d:
                        d[t] = d[t].copy()
                        d[t][varKey] = value
                    else:
                        d[t] = {varKey: value}
                else:
                    d[('DOVERD', dKey, overD)] = value

        return d


    def check_gradients(self, ins___dict):

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
                vars[('DOVERD', 'SUM___' + outKey, outKey)] = ones(vars[outKey].shape)
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
            approxGrad = approx_gradients(lambda v: sumOut({inKey: v}, outKey), ins___dict[inKey])
            check = check and allclose(approxGrad, vars[('DOVERD', outKey, inKey)], rtol = 1.e-3, atol = 1.e-6)
        for inKey, outKey in itertools.product(backwardInKeys, dSumKeys):
            approxGrad = approx_gradients(lambda v: sumOut({inKey: v}, outKey), ins___dict[inKey])
            check = check and allclose(approxGrad, vars[('DOVERD', 'SUM___' + outKey, inKey)],
                                       rtol = 1.e-3, atol = 1.e-6)

        return check



class Process:
    def __init__(self, *args, **kwargs):
        vars = set()
        steps = []
        for step in args:
            #print(step) ###########
            s = step_withDefaultForwardsAndBackwards(step)
            #print(s) ##############
            piece = s[0]
            for inVarTuple in piece.inKeys:
                vars.add(varName_fromVarTuple(inVarTuple))
            for outVarTuple in piece.outKeys:
                vars.add(varName_fromVarTuple(outVarTuple))
            steps += [s]
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
            self.steps += [s]

    def install(self, change_vars = {}):
        process = deepcopy(self)
        if change_vars:
            steps = []
            for step in process.steps:
                old_piece, old_forwardOutKeys, old_dKey_and_backwardInKeys = step
                new_piece = old_piece.install(change_vars = change_vars)
                new_forwardOutKeys = set()
                if old_forwardOutKeys:
                    for key in old_forwardOutKeys:
                        new_forwardOutKeys.add(change_var_in_piece_key(key, change_vars))
                if old_dKey_and_backwardInKeys:
                    old_dKey, old_backwardInKeys = old_dKey_and_backwardInKeys
                    new_dKey = change_var_in_piece_key(old_dKey, change_vars)
                    new_backwardInKeys = set()
                    for key in old_backwardInKeys:
                        new_backwardInKeys.add(change_var_in_piece_key(key, change_vars))
                    new_dKey_and_backwardInKeys = (new_dKey, new_backwardInKeys)
                else:
                    new_dKey_and_backwardInKeys = None
                steps += [new_piece, new_forwardOutKeys, new_dKey_and_backwardInKeys],

            ########print(steps) ##################
            process = Process(*steps)
        return process

    def run(self, dictObj, numTimes = 1, *args, **kwargs):
        d = dictObj.copy()
        for t in range(numTimes):
            for step in self.steps:
                piece, forwardOutKeys, dKey_and_backwardInKeys = step
                d = piece.run(d, forwardOutKeys, dKey_and_backwardInKeys)
        return d



def connect_processes(*args, **kwargs):
    p = deepcopy(args[0])
    for process in args[1:]:
        p.vars.update(process.vars)
        p.steps += process.steps
    return p



class Program:
    def __init__(self, pieces___dict, processes___dict):
        self.vars = set()
        self.pieces = pieces___dict
        self.processes = processes___dict
        for piece in pieces___dict.values():
            for key in (piece.inKeys).union(piece.outKeys):
                self.vars.add(varName_fromVarTuple(key))

    def install(self, from_old_var_names_to_new_var_names___dict):
        pieces = {}
        for piece_name, piece in self.pieces.items():
            pieces[piece_name] = piece.install(change_vars = from_old_var_names_to_new_var_names___dict)
        processes = {}
        for process_name, process in self.processes.items():
            processes[process_name] = process.install(change_vars = from_old_var_names_to_new_var_names___dict)
        return Program(pieces, processes)

    def run(self, dict_object, *args, **kwargs):
        d = dict_object.copy()
        for process_or_piece in args:
            if process_or_piece in self.processes:
                process = self.processes[process_or_piece]
                d = process.run(d)
            elif process_or_piece in self.pieces:
                piece = self.pieces[process_or_piece]
                d = piece.run(d, **kwargs)
        return d



class Project:
    def __init__(self, *args, **kwargs):
        self.vars = {}
        self.pieces = {}
        self.processes = {}
        self.programs = {}

    def add_variables(self, *args, **kwargs):
        self.variables.update(dict.fromkeys(args))
        self.variables.update(kwargs)

    def delete_variables(self, *args, **kwargs):
        for arg in args:
            del self.vars[arg]

    def set_variables(self, *args, **kwargs):
        for kwarg in kwargs:
            pass

    def getVar(self, *args):
        pass

    def run(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, tuple) and (arg[0] in self.programs):
                program = arg[0]
                process_or_piece_name = arg[1]
                if process_or_piece_name in self.programs[program].processes:
                    process = self.programs[program].processes[process_or_piece_name]
                    self.vars = process.run(self.vars)
                elif process_or_piece_name in self.programs[program].pieces:
                    piece = self.programs[program].pieces[process_or_piece_name]
                    self.vars = piece.run(self.vars, **kwargs)
            elif arg in self.processes:
                process = self.processes[arg]
                self.vars = process.run(self.vars)
            elif arg in self.pieces:
                piece = self.pieces[arg]
                self.vars = piece.run(self.vars, **kwargs)



def isOverD(varTuple):
    return (varTuple[0] == 'DOVERD') and (len(varTuple) == 2)



def isDOverD(varTuple):
    return (varTuple[0] == 'DOVERD') and (len(varTuple) == 3)



def varName_fromVarTuple(varTuple):
    # return variable name from tuple consisting of variable name and index/key
    if isinstance(varTuple, tuple):
        return varTuple[0]
    else:
        return varTuple



def change_var_in_piece_key(piece_key, from_old_vars_to_new_vars___dict):
    if isDOverD(piece_key):
        var_tuple_1 = piece_key[1]
        if isinstance(var_tuple_1, tuple) & (var_tuple_1[0] in from_old_vars_to_new_vars___dict):
            var_tuple_1[0] = from_old_vars_to_new_vars___dict[var_tuple_1[0]]
        elif var_tuple_1 in from_old_vars_to_new_vars___dict:
            var_tuple_1 = from_old_vars_to_new_vars___dict[var_tuple_1]
        var_tuple_2 = piece_key[2]
        if isinstance(var_tuple_2, tuple) & (var_tuple_2[0] in from_old_vars_to_new_vars___dict):
            var_tuple_2[0] = from_old_vars_to_new_vars___dict[var_tuple_2[0]]
        elif var_tuple_2 in from_old_vars_to_new_vars___dict:
            var_tuple_2 = from_old_vars_to_new_vars___dict[var_tuple_2]
        return ('DOVERD', var_tuple_1, var_tuple_2)
    elif isDOverD(piece_key):
        var_tuple = piece_key[1]
        if isinstance(var_tuple, tuple) & (var_tuple[0] in from_old_vars_to_new_vars___dict):
            return ('DOVERD', (from_old_vars_to_new_vars___dict[var_tuple[0]], var_tuple[1]))
        elif var_tuple in from_old_vars_to_new_vars___dict:
            return ('DOVERD', from_old_vars_to_new_vars___dict[var_tuple])
    elif isinstance(piece_key, tuple) & (piece_key[0] in from_old_vars_to_new_vars___dict):
        return (from_old_vars_to_new_vars___dict[piece_key[0]], piece_key[1])
    elif piece_key in from_old_vars_to_new_vars___dict:
        return from_old_vars_to_new_vars___dict[piece_key]
    else:
        return piece_key





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