from importlib import *
from math import *
from numpy import *
from legoML._common import *
from legoML.Classes import *
from legoML.funcs import *


f1 = [{'a1': 'i1', 'a2': 'i2'}, lambda a1, a2: a1 + a2, 's12']
f2 = [{'b1': 'i2', 'b2': 'i3'}, lambda b1, b2: b1 * b2, 'p23']
p = Piece(f1, f2)
d = {'i1': 3, 'i2': 4, 'i3': 5, 'i4': 6}
p.inKeys
p.func(d)
p1 = p.installPiece({'in1': 'i1', 'in2': 'i2', 'in3': 'i3'}, {'sum12': 's12', 'prod23': 'p23'})
p.inKeys
p.outKeys
p1.inKeys
d1 = {'in1': 3, 'in2': 4, 'in3': 5, 'in4': 6}
p1.func(d1)

d_toUpdate = {'a': [1, 2, 'c'], 'b': {'d': 10, 'e': 20}}
d_withValues = {'u': {'x': 100, 'y': 2000}, 'v': [123, 456, 'mnp']}
mapDict = {('a', 1): ('u', 'y'), ('b', 'e'): 'v'}
updateDictValues(d_toUpdate, mapDict, d_withValues)