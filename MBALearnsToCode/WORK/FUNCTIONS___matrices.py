from sympy import Symbol, symbols, sqrt
from sympy.matrices import MatrixSymbol, Matrix
from sympy.mpmath import sqrtm
from sympy.stats import sample, Normal, density
from numpy import array
from sympy import

mu = Symbol('mu')
sigma = Symbol('sigma', positive=True)

mu = MatrixSymbol('mu', 3, 1)
sigma = MatrixSymbol('sigma', 3, 3)
sigma.berkowitz()
x = Normal('x', mu, sigma)

iden = Matrix([[1, 0, 0],[0, 1, 0], [0, 0, mu ]])
sqr = sqrtm(iden)
ide = iden.as_immutable()
sqr.doit().evalf()

from scipy.stats import multivariate_normal


m1 = Matrix([[1, 2, 3]])
X = multivariate_normal(mean=[0, 0],
                        cov=[[1, 0], [0, 1]])

M = array([[0, 0],
           [0, 0],
           [0, 0]])
samples = X.rvs(10)

A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
assert A.cholesky() * A.cholesky().T == A