from frozendict import frozendict as fdict
from ProbabPy import DiscreteFinitePMF as PMF


def test___WALTER_2015___RoboAI___MidTerm_Q2():
    """test: WALTER (2015) "Planning, Learning & Estimation in Robotics & A.I.": Mid-Term Exam, Question 2"""

    p_B = PMF(dict.fromkeys(('B',)),
              {fdict(B=0): .7,
               fdict(B=1): .3})
    p_E = PMF(dict.fromkeys(('E',)),
              {fdict(E=0): .5,
               fdict(E=1): .5})
    p_A_on_E = PMF(dict.fromkeys(('A', 'E')),
                   {fdict(E=0, A=0): .9,
                    fdict(E=0, A=1): .1,
                    fdict(E=1, A=0): .1,
                    fdict(E=1, A=1): .9},
                   cond=dict(E=None))
    p_C_on_A_B = PMF(dict.fromkeys(('A', 'B', 'C')),
                     {fdict(A=0, B=0, C=0): .9,
                      fdict(A=0, B=0, C=1): .1,
                      fdict(A=0, B=1, C=0): .2,
                      fdict(A=0, B=1, C=1): .8,
                      fdict(A=1, B=0, C=0): .3,
                      fdict(A=1, B=0, C=1): .7,
                      fdict(A=1, B=1, C=0): .1,
                      fdict(A=1, B=1, C=1): .9},
                     cond=dict(A=None, B=None))
    p_D_on_C = PMF(dict.fromkeys(('C', 'D')),
                   {fdict(C=0, D=0): .6,
                    fdict(C=0, D=1): .4,
                    fdict(C=1, D=0): .4,
                    fdict(C=1, D=1): .6},
                   cond=dict(C=None))

    p_C_equal_1_on_A_equal_1 = (p_B * p_C_on_A_B.at(A=1, C=1)).marg('B')
    p_C_equal_1_on_A_equal_1.pprint()
    p_C_equal_1_on_A_equal_1___answer = PMF(dict.fromkeys(('A', 'C')),
                                            {fdict(C=1): .76},
                                            cond=dict(A=1),
                                            scope=dict(C=1))
    p_C_equal_1_on_A_equal_1___check = p_C_equal_1_on_A_equal_1.allclose(p_C_equal_1_on_A_equal_1___answer)

    p_A = (p_E * p_A_on_E).marg('E')
    p_A.pprint()
    p_A___answer = PMF(dict.fromkeys(('A',)),
                       {fdict(A=0): .5,
                        fdict(A=1): .5})
    p_A___check = p_A.allclose(p_A___answer)

    p_C_on_B_equal_1 = (p_A * p_C_on_A_B.at(B=1)).marg('A')
    p_C_on_B_equal_1.pprint()
    p_C_on_B_equal_1___answer = PMF(dict.fromkeys(('B', 'C')),
                                    {fdict(C=0): .15,
                                     fdict(C=1): .85},
                                    cond=dict(B=1))
    p_C_on_B_equal_1___check = p_C_on_B_equal_1.allclose(p_C_on_B_equal_1___answer)

    p_D_equal_1_on_B_equal_1 = (p_C_on_B_equal_1 * p_D_on_C.at(D=1)).marg('C')
    p_D_equal_1_on_B_equal_1.pprint()
    p_D_equal_1_on_B_equal_1___answer = PMF(dict.fromkeys(('B', 'D')),
                                            {fdict(D=1): .57},
                                            cond=dict(B=1),
                                            scope=dict(D=1))
    p_D_equal_1_on_B_equal_1___check = p_D_equal_1_on_B_equal_1.allclose(p_D_equal_1_on_B_equal_1___answer)

    p_C = (p_A * p_B * p_C_on_A_B).marg('A', 'B')
    p_C.pprint()
    p_C___answer = PMF(dict.fromkeys(('C',)),
                       {fdict(C=0): .9 * .5 * .7 + .2 * .5 * .3 + .3 * .5 * .7 + .1 * .5 * .3,
                        fdict(C=1): .1 * .5 * .7 + .8 * .5 * .3 + .7 * .5 * .7 + .9 * .5 * .3})
    p_C___check = p_C.allclose(p_C___answer)

    p_D_equal_1 = (p_C * p_D_on_C.at(D=1)).marg('C')
    p_D_equal_1.pprint()
    p_D_equal_1___answer = PMF(dict.fromkeys(('D',)),
                               {fdict(D=1): .4 * (.9 * .5 * .7 + .2 * .5 * .3 + .3 * .5 * .7 + .1 * .5 * .3) +
                                            .6 * (.1 * .5 * .7 + .8 * .5 * .3 + .7 * .5 * .7 + .9 * .5 * .3)},
                               scope=dict(D=1))
    p_D_equal_1___check = p_D_equal_1.allclose(p_D_equal_1___answer)

    p_A_C_on_B = p_A * p_C_on_A_B
    p_C_on_B = p_A_C_on_B.marg('A')
    p_B_C_D = p_B * p_C_on_B * p_D_on_C
    p_B_D = p_B_C_D.marg('C')
    p_B_equal_1_on_D_equal_1 = p_B_D.cond(D=1).norm().at(B=1)
    p_B_equal_1_on_D_equal_1.pprint()
    p_B_equal_1_on_D_equal_1___answer = PMF(dict.fromkeys(('B', 'D')),
                                            {fdict(B=1): .337},
                                            cond=dict(D=1),
                                            scope=dict(B=1))
    p_B_equal_1_on_D_equal_1___check = p_B_equal_1_on_D_equal_1.allclose(p_B_equal_1_on_D_equal_1___answer, atol=1e-3)

    assert p_C_equal_1_on_A_equal_1___check & p_A___check & p_C_on_B_equal_1___check &\
        p_D_equal_1_on_B_equal_1___check & p_C___check & p_D_equal_1___check & p_B_equal_1_on_D_equal_1___check


if __name__ == '__main__':
    test___WALTER_2015___RoboAI___MidTerm_Q2()
