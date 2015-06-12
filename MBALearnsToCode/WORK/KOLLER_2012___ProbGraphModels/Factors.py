from __future__ import print_function
from frozen_dict import FrozenDict as fdict
from sympy import log
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import\
    discrete_finite_mass_function as pmf, discrete_finite_mass_functions_all_close as pmf_allclose


def UNIT_TEST___KOLLER_2012___ProbGraphModels___Factors():

    f = pmf(dict.fromkeys(('a', 'b')),
            dict(mappings={fdict(a=1, b=1): -log(.5),
                           fdict(a=1, b=2): -log(.8),
                           fdict(a=2, b=1): -log(.1),
                           fdict(a=2, b=2): -log(.0),
                           fdict(a=3, b=1): -log(.3),
                           fdict(a=3, b=2): -log(.9)}))

    f_1 = pmf(dict.fromkeys(('b', 'c')),
              dict(mappings={fdict(b=1, c=1): -log(.5),
                             fdict(b=1, c=2): -log(.7),
                             fdict(b=2, c=1): -log(.1),
                             fdict(b=2, c=2): -log(.2)}))
    f *= f_1
    f.pprint()
    f___answer = pmf(dict.fromkeys(('a', 'b', 'c')),
                     dict(mappings={fdict(a=1, b=1, c=1): -log(.25),
                                    fdict(a=1, b=1, c=2): -log(.35),
                                    fdict(a=1, b=2, c=1): -log(.08),
                                    fdict(a=1, b=2, c=2): -log(.16),
                                    fdict(a=2, b=1, c=1): -log(.05),
                                    fdict(a=2, b=1, c=2): -log(.07),
                                    fdict(a=2, b=2, c=1): -log(.0),
                                    fdict(a=2, b=2, c=2): -log(.0),
                                    fdict(a=3, b=1, c=1): -log(.15),
                                    fdict(a=3, b=1, c=2): -log(.21),
                                    fdict(a=3, b=2, c=1): -log(.09),
                                    fdict(a=3, b=2, c=2): -log(.18)}))
    f___check = pmf_allclose(f, f___answer)
    print('Factor Multiplication check:', f___check)

    m = f.marginalize('b')
    m.pprint()
    m___answer = pmf(dict.fromkeys(('a', 'c')),
                     dict(mappings={fdict(a=1, c=1): -log(.33),
                                    fdict(a=1, c=2): -log(.51),
                                    fdict(a=2, c=1): -log(.05),
                                    fdict(a=2, c=2): -log(.07),
                                    fdict(a=3, c=1): -log(.24),
                                    fdict(a=3, c=2): -log(.39)}))
    m___check = pmf_allclose(m, m___answer)
    print('Factor Marginalization check:', m___check)

    return f___check & m___check
