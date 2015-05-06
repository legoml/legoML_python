from numpy import array, squeeze
from sympy import Symbol, symbols, oo, pi, exp
from sympy.matrices import MatrixSymbol, BlockMatrix, block_collapse, eye, Matrix
from sympy.matrices.expressions.blockmatrix import bc_matadd
from sympy.integrals import integrate
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from MBALearnsToCode.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor
from MBALearnsToCode.Functions.FUNCTIONS___matrices import\
    sympy_matrix_determinant_expansion, sympy_matrix_inverse_expansion

x, y, m, mx, my, s, sx, sy, sxy = symbols('x y m mx my s sx sy sxy')

nx, ny = symbols('nx ny')

x = MatrixSymbol('x', 5, 1)
y = MatrixSymbol('y', 5, 1)
B = BlockMatrix([[x, x], [y, y]])

d = ((2 * pi) ** (-1 / 2)) * (1 ** (-1 / 2)) * exp(- (x - 0) * (1 ** -1) * (x - 0) / 2)
V = array([[x], [y]], dtype=object)
M = array([[m], [m]], dtype=object)
S = array([[sx, 0], [0, sy]], dtype=object)

B = BlockMatrix([])

d = sympy_normal_density(V, M, S)
r = integrate(d, (x, -oo, oo), manual=True)
s = integrate(r, (y, -oo, oo), manual=True)
f = Factor(d, scope={'x', 'y'})


V = array([[x], [y]], dtype=object)
M = array([[mx], [my]], dtype=object)
S = array([[sx, sxy], [sxy, sy]], dtype=object)
g = f.eliminate((('x', 'integrate', (-oo, oo)),))
h = f.eliminate((('x', 'integrate', (-oo, oo)),))


sympy_matrix_determinant_expansion(M)
sympy_matrix_inverse_expansion(M)



f1 = Factor(DFDF({fdict(a=1, b=1): 0.5,
                  fdict(a=1, b=2): 0.8,
                  fdict(a=2, b=1): 0.1,
                  fdict(a=3, b=1): 0.3,
                  fdict(a=3, b=2): 0.9}))
f2 = Factor(DFDF({fdict(b=1, c=1): 0.5,
                  fdict(b=1, c=2): 0.7,
                  fdict(b=2, c=1): 0.1,
                  fdict(b=2, c=2): 0.2}))
f = f1.multiply(f2)

g = f.eliminate((('b', 'sum', (1, 2)),))
g.pprint()

from


d
import theano


import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']="C:/Programs/spark-1.3.1-bin-hadoop2.6"

# Append pyspark  to Python Path
sys.path.append("C:\Programs\spark-1.3.1-bin-hadoop2.6\python")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

SparkContext