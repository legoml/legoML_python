from numpy import array
from sympy import Symbol, symbols, sympify, pi, exp
from sympy.matrices import Matrix, MatrixSymbol, BlockMatrix, det
from MBALearnsToCode.Functions.FUNCTIONS___sympy import sympy_xreplace_doit_explicit_evalf

# declare matrices of various sizes
na, nb, nc, nd, nx, ny, nz, nw = symbols('na nb nc nd nx ny nz nw')
a = MatrixSymbol('a', na, 1)
b = MatrixSymbol('b', nb, 1)
c = MatrixSymbol('c', nc, 1)
d = MatrixSymbol('d', nd, 1)
x = MatrixSymbol('x', na, 1)
y = MatrixSymbol('y', nb, 1)
z = MatrixSymbol('z', nc, 1)
w = MatrixSymbol('w', nd, 1)

real_a = array([[1], [2]])
real_b = array([[10], [20], [30]])


# declare Block Matrix

B = BlockMatrix([[a, a], [b, b]])
C = B * B.T
D = sympy_xreplace_doit_explicit_evalf(C, {a: Matrix(real_a), b: Matrix(real_b)})
D = C.xreplace({a: Matrix(real_a), b: Matrix(real_b)})#.doit(deep=True).as_explicit()
D0 = D.args[0]
D1 = D.args[1]

x = MatrixSymbol('x', nx, 1)
y = MatrixSymbol('y', ny, 1)
mx = MatrixSymbol('mx', nx, 1)
my = MatrixSymbol('my', ny, 1)
Sx = MatrixSymbol('Sx', nx, nx)
Sy = MatrixSymbol('Sy', ny, ny)
Sxy = MatrixSymbol('Sxy', nx, ny)

X = BlockMatrix([[x], [y]])
M = BlockMatrix([[mx], [my]])
S = BlockMatrix([[Sx, Sxy], [Sxy.T, Sy]])
d = nx + ny
sqerr = - (X - M).transpose() * S.inverse() * (X - M) / 2

dense = ((2 * pi) ** (-d / 2)) * (det(S) ** (-1 / 2)) * exp(det(- (X - M).transpose() * S.inverse() * (X - M) / 2))