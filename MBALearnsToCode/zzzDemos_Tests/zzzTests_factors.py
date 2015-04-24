from sympy import Symbol, NumberSymbol, MatrixSymbol
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from MBALearnsToCode.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor
from pprint import pprint


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
f.print_factor()
# expected result:
# (('a', 1), ('b', 1), ('c', 1)): 0.25
# (('a', 1), ('b', 1), ('c', 2)): 0.35
# (('a', 1), ('b', 2), ('c', 1)): 0.08
# (('a', 1), ('b', 2), ('c', 2)): 0.16
# (('a', 2), ('b', 1), ('c', 1)): 0.05
# (('a', 2), ('b', 1), ('c', 2)): 0.07
# (('a', 2), ('b', 2), ('c', 1)): 0
# (('a', 2), ('b', 2), ('c', 2)): 0
# (('a', 3), ('b', 1), ('c', 1)): 0.15
# (('a', 3), ('b', 1), ('c', 2)): 0.21
# (('a', 3), ('b', 1), ('c', 2)): 0.21
# (('a', 3), ('b', 2), ('c', 1)): 0.09
# (('a', 3), ('b', 2), ('c', 2)): 0.18

g = f.eliminate((('b', 'sum', (1, 2)),))
g.print_factor()


x = Symbol('x')
y = Symbol('y')

dfdf = DFDF({fdict({('a', 1): 10, ('b', 2): 20}): x * y,
             fdict({('a', 1): 100, ('b', 2): 200}): 300})
pprint(dfdf.args)
pprint(dfdf.discrete_finite_mappings)
pprint(dfdf.subs({('a', 1): 10, ('b', 2): 20, x: 3}).discrete_finite_mappings)
pprint(dfdf.subs({('a', 1): 100, ('b', 2): 200, x: 3}).discrete_finite_mappings)
pprint(dfdf.subs({('a', 1): 100, ('b', 2): 200}).discrete_finite_mappings)

s = Factor(x * y)


from lshash import LSHash
lsh = LSHash(6, 8)
lsh.index([1,2,3,4,5,6,7,8])
lsh.index([2,3,4,5,6,7,8,9])
lsh.index([10,12,99,1,5,31,2,3])
lsh.query([1,2,3,4,5,6,7,7])