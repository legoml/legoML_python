class Flow:

    def __init__(self, func):
        self.ins = {}
        self.func = func
        self.outs = {}

    def flow(self, ins):
        self.ins = ins
        self.outs = self.func(**ins)

    def connect(self, flow1, flow2):
        self.ins = flow1.ins
        self.func = lambda x: flow2.func(flow1.func(x))