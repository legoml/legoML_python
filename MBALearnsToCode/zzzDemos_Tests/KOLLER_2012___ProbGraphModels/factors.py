# TEST factor_product
# (examples from Coursera: "Probabilistic Graphical Models" (Daphne Koller)
from sympy import log
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from MBALearnsToCode.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor
from MBALearnsToCode.UserDefinedClasses.CLASSES___ProbabilityDensityFunctions import\
    discrete_finite_mass_function as dfmf

m1 = dfmf(dict.fromkeys(('a', 'b')),
          dict(mappings={fdict(a=1, b=1): -log(0.5),
                         fdict(a=1, b=2): -log(0.8),
                         fdict(a=2, b=1): -log(0.1),
                         fdict(a=3, b=1): -log(0.3),
                         fdict(a=3, b=2): -log(0.9)}))

m2 = dfmf(dict.fromkeys(('b', 'c')),
          dict(mappings={fdict(b=1, c=1): -log(0.5),
                         fdict(b=1, c=2): -log(0.7),
                         fdict(b=2, c=1): -log(0.1),
                         fdict(b=2, c=2): -log(0.2)}))

m = m1.multiply(m2)

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
f.pprint()
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
g.pprint()