from theano import function, SymbolicOutput, Out
from timeit import timeit
from sympy import symbols, sympify, det
from sympy.matrices import BlockMatrix, MatrixSymbol
from sympy.printing.theanocode import theano_function

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

z = sympify(1.) + sympify(1.)