from theano import function, SymbolicOutput, Out
from timeit import timeit
from sympy import symbols, sympify, det, log
from sympy.matrices import BlockMatrix, MatrixSymbol
from sympy.printing.theanocode import theano_function
from numpy import array


from sympy.matrices import MatrixSymbol, BlockMatrix, Matrix

x = MatrixSymbol('x', 2, 2)
b = BlockMatrix([[x, x]])

from numpy import array
a = array([[1, 2],
           [3, 4]])

m = BlockMatrix([i for i in range(5)])

o = log(2) + log(3)
f = theano_function([], o)

from frozendict import frozendict


from MathFunc import MathFunc

d0 = MathFunc(dict.fromkeys(('a', 'b')),
             {frozendict(a=1, b=2): 3,
              frozendict(a=10, b=20): 30})

d = MathFunc(dict.fromkeys(('a', 'b')),
             {frozendict(a=1, b=2): 3,
              frozendict(a=10, b=20): 30})
d1 = MathFunc(dict.fromkeys(('b', 'c')),
             {frozendict(c=3, b=2): 30,
              frozendict(c=30, b=20): 300})
d * 2

x, y = symbols('x y')
f = theano_function([x], [y-y], on_unused_input='ignore')
z = x + y
X = MatrixSymbol('X', 3, 2)
Y = MatrixSymbol('Y', 5, 2)
Z = BlockMatrix([[X], [Y]])

f = theano_function([X, Y], [Z])

f([[1, 1],
   [1, 1],
   [1, 1]],
  [[1, 1],
   [1, 1],
   [1, 1],
   [1, 1],
   [1, 1]])

W = BlockMatrix([[X, X]])
g = theano_function([X], [W])
g([[1, 1],
   [1, 1],
   [1, 1]])
g(X)

h = theano_function([x, y], [z])

U = MatrixSymbol('U', 3, 3)
h = theano_function([U], [det(U)])
timeit(h([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))


g_a = g.at(dict(w=Matrix(w_value)))
g_a.pprint()

g_max = g.max()
g_max.pprint()

g_a_max = g_a.max()
g_a_max.pprint()

g_m_wz = g.marginalize(('w', 'z'))
g_m_wz.pprint()

g_c = g.condition(dict(w=Matrix(w_value)))
g_c.pprint()

g_one = g_m = g.marginalize(('x', 'y', 'z', 'w'))
g_one.pprint()