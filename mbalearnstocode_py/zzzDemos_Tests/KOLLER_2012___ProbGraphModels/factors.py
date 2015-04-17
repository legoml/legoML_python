# TEST factor_product
# (examples from Coursera: "Probabilistic Graphical Models" (Daphne Koller)

from frozen_dict import FrozenDict as fdict
from MBALearnsToCode_Py.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from MBALearnsToCode_Py.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor


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
f.print()
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
g.print()