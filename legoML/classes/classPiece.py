from legoML._common import itself, subDict_withNewKeys, func_onDict, dictOfFuncs



class Piece:
    def __init__(self, *args, **kwargs):
        inKeys = set()
        funcs___dict = {}
        outKeys = set()
        for eachList in args:
            argKeysFromInKeys_thisList___dict, func_thisList, outKey_thisList = eachList
            inKeys.update(argKeysFromInKeys_thisList___dict.values())
            funcs___dict[outKey_thisList] = func_onDict(func_thisList, argKeysFromInKeys_thisList___dict)
            outKeys.add(outKey_thisList)
        self.inKeys = inKeys
        self.func = lambda ins___dict: dictOfFuncs(funcs___dict, ins___dict)
        self.outKeys = outKeys



#def connectPieces(pieces___list, inKeysFromOutKeys___listOfDicts):
#    piece = pieces___list[0]
#    del pieces___list[0]
#    for i in range(len(pieces___list)):
#        nextPiece = pieces___list[i]
#        inKeysFromOutKeys___dict = inKeysFromOutKeys___listOfDicts[i]
#        #func_connect = lambda d: subDict_withNewKeys(inKeysFromOutKeys___dict, d)
#        #piece.func = lambda ins___dict: func_connect(piece.func(ins___dict))
#        #piece.func = lambda ins___dict: nextPiece.func(piece.func(ins___dict))
#        piece.outKeys = nextPiece.outKeys
#    return piece