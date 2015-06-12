from __future__ import print_function
from frozen_dict import FrozenDict as fdict
from sympy import log
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import\
    discrete_finite_mass_function as pmf, discrete_finite_mass_functions_all_close as pmf_allclose


def UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q1():

    p_C = pmf(dict.fromkeys(('C',)),
              dict(mappings={fdict(C='a'): -log(.3),
                             fdict(C='n'): -log(.5),
                             fdict(C='l'): -log(.2)}))
    p_T1_on_C = pmf(dict.fromkeys(('C', 'T1')),
                    dict(mappings={fdict(C='a', T1=0): -log(.3),
                                   fdict(C='a', T1=1): -log(.7),
                                   fdict(C='n', T1=0): -log(.5),
                                   fdict(C='n', T1=1): -log(.5),
                                   fdict(C='l', T1=0): -log(.8),
                                   fdict(C='l', T1=1): -log(.2)}),
                    conditions=dict(C=None))
    p_T2_on_C = pmf(dict.fromkeys(('C', 'T2')),
                    dict(mappings={fdict(C='a', T2=0): -log(.2),
                                   fdict(C='a', T2=1): -log(.8),
                                   fdict(C='n', T2=0): -log(.6),
                                   fdict(C='n', T2=1): -log(.4),
                                   fdict(C='l', T2=0): -log(.9),
                                   fdict(C='l', T2=1): -log(.1)}),
                    conditions=dict(C=None))
    p_D_on_T1_T2 = pmf(dict.fromkeys(('T1', 'T2', 'D')),
                       dict(mappings={fdict(T1=0, T2=0, D=0): -log(1.),
                                      fdict(T1=0, T2=0, D=1): -log(.0),
                                      fdict(T1=0, T2=1, D=0): -log(1.),
                                      fdict(T1=0, T2=1, D=1): -log(.0),
                                      fdict(T1=1, T2=0, D=0): -log(.0),
                                      fdict(T1=1, T2=0, D=1): -log(1.),
                                      fdict(T1=1, T2=1, D=0): -log(.0),
                                      fdict(T1=1, T2=1, D=1): -log(1.)}),
                       conditions=dict(T1=None, T2=None))

    p_T1 = (p_C * p_T1_on_C).marginalize('C')
    p_T1.pprint()
    p_T1___answer = pmf(dict.fromkeys(('T1',)),
                        dict(mappings={fdict(T1=0): -log(.5),
                                       fdict(T1=1): -log(.5)}))
    p_T1___check = pmf_allclose(p_T1, p_T1___answer)

    p_T2 = (p_C * p_T2_on_C).marginalize('C')
    p_T2.pprint()
    p_T2___answer = pmf(dict.fromkeys(('T2',)),
                        dict(mappings={fdict(T2=0): -log(.54),
                                       fdict(T2=1): -log(.46)}))
    p_T2___check = pmf_allclose(p_T2, p_T2___answer)

    p_T1_T2 = (p_C * p_T1_on_C * p_T2_on_C).marginalize('C')
    p_T1_T2.pprint()
    p_T1_T2___answer = pmf(dict.fromkeys(('T1', 'T2')),
                           dict(mappings={fdict(T1=0, T2=0): -log(.312),
                                          fdict(T1=0, T2=1): -log(.188),
                                          fdict(T1=1, T2=0): -log(.228),
                                          fdict(T1=1, T2=1): -log(.272)}))
    p_T1_T2___check = pmf_allclose(p_T1_T2, p_T1_T2___answer)

    p_T1_p_T2 = p_T1 * p_T2
    p_T1_p_T2.pprint()

    p_T1_T2_on_C_equal_n = (p_C * p_T1_on_C * p_T2_on_C).condition(C='n').normalize()
    p_T1_T2_on_C_equal_n.pprint()
    p_T1_T2_on_C_equal_n___answer = pmf(dict.fromkeys(('C', 'T1', 'T2')),
                                        dict(mappings={fdict(T1=0, T2=0): -log(.3),
                                                       fdict(T1=0, T2=1): -log(.2),
                                                       fdict(T1=1, T2=0): -log(.3),
                                                       fdict(T1=1, T2=1): -log(.2)}),
                                        conditions=dict(C='n'))
    p_T1_T2_on_C_equal_n___check = pmf_allclose(p_T1_T2_on_C_equal_n, p_T1_T2_on_C_equal_n___answer)

    p_T1_on_C_equal_n = p_T1_on_C.at(C='n')
    p_T1_on_C_equal_n.pprint()
    p_T2_on_C_equal_n = p_T2_on_C.at(C='n')
    p_on_T1_T2_and_D_equal_1 = p_D_on_T1_T2.condition(D=1)
    p_T1_T2_on_C_equal_n_and_D_equal_1 = (p_T1_on_C_equal_n * p_T2_on_C_equal_n * p_on_T1_T2_and_D_equal_1).normalize()
    p_T1_T2_on_C_equal_n_and_D_equal_1.pprint()

    p_T1_T2_on_C_equal_n_and_D_equal_1___alternative =\
        (p_C * p_T1_on_C * p_T2_on_C * p_D_on_T1_T2).condition(C='n', D=1).normalize()
    p_T1_T2_on_C_equal_n_and_D_equal_1___alternative.pprint()

    p_T1_T2_on_C_equal_n_and_D_equal_1___answer = pmf(dict.fromkeys(('C', 'T1', 'T2', 'D')),
                                                      dict(mappings={fdict(T1=0, T2=0): -log(.0),
                                                                     fdict(T1=0, T2=1): -log(.0),
                                                                     fdict(T1=1, T2=0): -log(.6),
                                                                     fdict(T1=1, T2=1): -log(.4)}),
                                                      conditions=dict(C='n', D=1))
    p_T1_T2_on_C_equal_n_and_D_equal_1___answer.pprint()
    p_T1_T2_on_C_equal_n_and_D_equal_1___check = pmf_allclose(p_T1_T2_on_C_equal_n_and_D_equal_1,
                                                              p_T1_T2_on_C_equal_n_and_D_equal_1___alternative,
                                                              p_T1_T2_on_C_equal_n_and_D_equal_1___answer)

    return p_T1___check & p_T2___check & p_T1_T2___check & p_T1_T2_on_C_equal_n___check &\
        p_T1_T2_on_C_equal_n_and_D_equal_1___check