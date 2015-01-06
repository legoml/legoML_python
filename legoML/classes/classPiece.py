class Piece:

    def __init__(self, inKeys, func, outKeys):
        self.ins = dict.fromkeys(inKeys)
        self.func = func
        self.outs = dict.fromkeys(outKeys)