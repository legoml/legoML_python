from __future__ import print_function
from sympy import log
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import\
    discrete_finite_mass_function as dfmf, discrete_finite_mass_functions_all_close as dfmf_allclose


def UNIT_TEST___KOLLER_2012___ProbGraphModels___Factors():

    f_1 = dfmf(dict.fromkeys(('a', 'b')),
               dict(mappings={fdict(a=1, b=1): -log(0.5),
                              fdict(a=1, b=2): -log(0.8),
                              fdict(a=2, b=1): -log(0.1),
                              fdict(a=2, b=2): -log(0.0),
                              fdict(a=3, b=1): -log(0.3),
                              fdict(a=3, b=2): -log(0.9)}))

    f_2 = dfmf(dict.fromkeys(('b', 'c')),
               dict(mappings={fdict(b=1, c=1): -log(0.5),
                              fdict(b=1, c=2): -log(0.7),
                              fdict(b=2, c=1): -log(0.1),
                              fdict(b=2, c=2): -log(0.2)}))
    f = f_1.multiply(f_2)
    f.pprint()
    f___answer = dfmf(dict.fromkeys(('a', 'b', 'c')),
                      dict(mappings={fdict(a=1, b=1, c=1): -log(0.25),
                                     fdict(a=1, b=1, c=2): -log(0.35),
                                     fdict(a=1, b=2, c=1): -log(0.08),
                                     fdict(a=1, b=2, c=2): -log(0.16),
                                     fdict(a=2, b=1, c=1): -log(0.05),
                                     fdict(a=2, b=1, c=2): -log(0.07),
                                     fdict(a=2, b=2, c=1): -log(0.0),
                                     fdict(a=2, b=2, c=2): -log(0.0),
                                     fdict(a=3, b=1, c=1): -log(0.15),
                                     fdict(a=3, b=1, c=2): -log(0.21),
                                     fdict(a=3, b=2, c=1): -log(0.09),
                                     fdict(a=3, b=2, c=2): -log(0.18)}))
    f___check = dfmf_allclose(f, f___answer)
    print('Factor Multiplication check:', f___check)

    m = f.marginalize(('b',))
    m.pprint()
    m___answer = dfmf(dict.fromkeys(('a', 'c')),
                      dict(mappings={fdict(a=1, c=1): -log(0.33),
                                     fdict(a=1, c=2): -log(0.51),
                                     fdict(a=2, c=1): -log(0.05),
                                     fdict(a=2, c=2): -log(0.07),
                                     fdict(a=3, c=1): -log(0.24),
                                     fdict(a=3, c=2): -log(0.39)}))
    m___check = dfmf_allclose(m, m___answer)
    print('Factor Marginalization check:', m___check)

    return f___check & m___check
