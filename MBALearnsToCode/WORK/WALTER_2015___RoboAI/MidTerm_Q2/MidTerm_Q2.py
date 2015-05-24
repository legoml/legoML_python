from __future__ import print_function
from sympy import log
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import\
    discrete_finite_mass_function as dfmf, discrete_finite_mass_functions_all_close as dfmf_allclose


def UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q2():

    p_B = dfmf(dict.fromkeys(('B',)),
               dict(mappings={fdict(B=0): -log(.7),
                              fdict(B=1): -log(.3)}))
    p_E = dfmf(dict.fromkeys(('E',)),
               dict(mappings={fdict(E=0): -log(.5),
                              fdict(E=1): -log(.5)}))
    p_A_on_E = dfmf(dict.fromkeys(('A', 'E')),
                    dict(mappings={fdict(E=0, A=0): -log(.9),
                                   fdict(E=0, A=1): -log(.1),
                                   fdict(E=1, A=0): -log(.1),
                                   fdict(E=1, A=1): -log(.9)}),
                    conditions=dict(E=None))
    p_C_on_A_B = dfmf(dict.fromkeys(('A', 'B', 'C')),
                      dict(mappings={fdict(A=0, B=0, C=0): -log(.9),
                                     fdict(A=0, B=0, C=1): -log(.1),
                                     fdict(A=0, B=1, C=0): -log(.2),
                                     fdict(A=0, B=1, C=1): -log(.8),
                                     fdict(A=1, B=0, C=0): -log(.3),
                                     fdict(A=1, B=0, C=1): -log(.7),
                                     fdict(A=1, B=1, C=0): -log(.1),
                                     fdict(A=1, B=1, C=1): -log(.9)}),
                      conditions=dict(A=None, B=None))
    p_D_on_C = dfmf(dict.fromkeys(('C', 'D')),
                    dict(mappings={fdict(C=0, D=0): -log(.6),
                                   fdict(C=0, D=1): -log(.4),
                                   fdict(C=1, D=0): -log(.4),
                                   fdict(C=1, D=1): -log(.6)}),
                    conditions=dict(C=None))

    p_C_equal_1_on_A_equal_1 = (p_B.multiply(p_C_on_A_B.at(dict(A=1, C=1)))).marginalize(('B',))
    p_C_equal_1_on_A_equal_1.pprint()
    p_C_equal_1_on_A_equal_1___answer = dfmf(dict.fromkeys(('A', 'C')),
                                     dict(mappings={fdict(C=1): -log(.76)}),
                                     conditions=dict(A=1),
                                     scope=dict(C=1))
    p_C_equal_1_on_A_equal_1___check = dfmf_allclose(p_C_equal_1_on_A_equal_1, p_C_equal_1_on_A_equal_1___answer)

    p_A = (p_E.multiply(p_A_on_E)).marginalize(('E',))
    p_A.pprint()
    p_A___answer = dfmf(dict.fromkeys(('A',)),
                        dict(mappings={fdict(A=0): -log(.5),
                                       fdict(A=1): -log(.5)}))
    p_A___check = dfmf_allclose(p_A, p_A___answer)

    p_C_on_B_equal_1 = (p_A.multiply(p_C_on_A_B.at(dict(B=1)))).marginalize(('A',))
    p_C_on_B_equal_1.pprint()
    p_C_on_B_equal_1___answer = dfmf(dict.fromkeys(('B', 'C')),
                                     dict(mappings={fdict(C=0): -log(.15),
                                                    fdict(C=1): -log(.85)}),
                                     conditions=dict(B=1))
    p_C_on_B_equal_1___check = dfmf_allclose(p_C_on_B_equal_1, p_C_on_B_equal_1___answer)

    p_D_equal_1_on_B_equal_1 = (p_C_on_B_equal_1.multiply(p_D_on_C.at(dict(D=1)))).marginalize(('C',))
    p_D_equal_1_on_B_equal_1.pprint()
    p_D_equal_1_on_B_equal_1___answer = dfmf(dict.fromkeys(('B', 'D')),
                                             dict(mappings={fdict(D=1): -log(.57)}),
                                             conditions=dict(B=1),
                                             scope=dict(D=1))
    p_D_equal_1_on_B_equal_1___check = dfmf_allclose(p_D_equal_1_on_B_equal_1, p_D_equal_1_on_B_equal_1___answer)

    p_C = p_A.multiply(p_B, p_C_on_A_B).marginalize(('A', 'B'))
    p_C.pprint()
    p_C___answer = dfmf(dict.fromkeys(('C')),
                        dict(mappings={fdict(C=0): -log(.9 * .5 * .7 + .2 * .5 * .3 + .3 * .5 * .7 + .1 * .5 * .3),
                                       fdict(C=1): -log(.1 * .5 * .7 + .8 * .5 * .3 + .7 * .5 * .7 + .9 * .5 * .3)}))
    p_C___check = dfmf_allclose(p_C, p_C___answer)

    p_D_equal_1 = p_C.multiply(p_D_on_C.at(dict(D=1))).marginalize(('C',))
    p_D_equal_1.pprint()
    p_D_equal_1___answer = dfmf(dict.fromkeys(('D')),
                                dict(mappings={fdict(D=1): -log(.4 * (.9 * .5 * .7 + .2 * .5 * .3 +
                                                                      .3 * .5 * .7 + .1 * .5 * .3) +
                                                                .6 * (.1 * .5 * .7 + .8 * .5 * .3 +
                                                                      .7 * .5 * .7 + .9 * .5 * .3))}),
                                scope=dict(D=1))
    p_D_equal_1___check = dfmf_allclose(p_D_equal_1, p_D_equal_1___answer)

    p_A_C_on_B = p_A.multiply(p_C_on_A_B)
    p_C_on_B = p_A_C_on_B.marginalize(('A',))
    p_B_C_D = p_B.multiply(p_C_on_B, p_D_on_C)
    p_B_D = p_B_C_D.marginalize(('C',))
    p_B_equal_1_on_D_equal_1 = p_B_D.condition(dict(D=1)).normalize().at(dict(B=1))
    p_B_equal_1_on_D_equal_1.pprint()
    p_B_equal_1_on_D_equal_1___answer = dfmf(dict.fromkeys(('B', 'D')),
                                             dict(mappings={fdict(B=1): -log(.337)}),
                                             conditions=dict(D=1),
                                             scope=dict(B=1))
    p_B_equal_1_on_D_equal_1___check = dfmf_allclose(p_B_equal_1_on_D_equal_1, p_B_equal_1_on_D_equal_1___answer,
                                                     atol=1e-3)

    return p_C_equal_1_on_A_equal_1___check & p_A___check & p_C_on_B_equal_1___check &\
        p_D_equal_1_on_B_equal_1___check & p_C___check & p_D_equal_1___check & p_B_equal_1_on_D_equal_1___check