from frozendict import frozendict as fdict
from ProbabPy import DiscreteFinitePMF as PMF


def test___WALTER_2015___RoboAI___ProbSet1_Q1():
    """test: WALTER (2015) "Planning, Learning & Estimation in Robotics & A.I.": Problem Set 1, Question 1"""
    p_C = PMF(dict.fromkeys(('C',)),
              {fdict(C='a'): .3,
               fdict(C='n'): .5,
               fdict(C='l'): .2})
    p_T1_on_C = PMF(dict.fromkeys(('C', 'T1')),
                    {fdict(C='a', T1=0): .3,
                     fdict(C='a', T1=1): .7,
                     fdict(C='n', T1=0): .5,
                     fdict(C='n', T1=1): .5,
                     fdict(C='l', T1=0): .8,
                     fdict(C='l', T1=1): .2},
                    cond=dict(C=None))
    p_T2_on_C = PMF(dict.fromkeys(('C', 'T2')),
                    {fdict(C='a', T2=0): .2,
                     fdict(C='a', T2=1): .8,
                     fdict(C='n', T2=0): .6,
                     fdict(C='n', T2=1): .4,
                     fdict(C='l', T2=0): .9,
                     fdict(C='l', T2=1): .1},
                    cond=dict(C=None))
    p_D_on_T1_T2 = PMF(dict.fromkeys(('T1', 'T2', 'D')),
                       {fdict(T1=0, T2=0, D=0): 1.,
                        fdict(T1=0, T2=0, D=1): .0,
                        fdict(T1=0, T2=1, D=0): 1.,
                        fdict(T1=0, T2=1, D=1): .0,
                        fdict(T1=1, T2=0, D=0): .0,
                        fdict(T1=1, T2=0, D=1): 1.,
                        fdict(T1=1, T2=1, D=0): .0,
                        fdict(T1=1, T2=1, D=1): 1.},
                       cond=dict(T1=None, T2=None))

    p_T1 = (p_C * p_T1_on_C).marg('C')
    p_T1.pprint()
    p_T1___answer = PMF(dict.fromkeys(('T1',)),
                        {fdict(T1=0): .5,
                         fdict(T1=1): .5})
    p_T1___check = p_T1.allclose(p_T1___answer)

    p_T2 = (p_C * p_T2_on_C).marg('C')
    p_T2.pprint()
    p_T2___answer = PMF(dict.fromkeys(('T2',)),
                        {fdict(T2=0): .54,
                         fdict(T2=1): .46})
    p_T2___check = p_T2.allclose(p_T2___answer)

    p_T1_T2 = (p_C * p_T1_on_C * p_T2_on_C).marg('C')
    p_T1_T2.pprint()
    p_T1_T2___answer = PMF(dict.fromkeys(('T1', 'T2')),
                           {fdict(T1=0, T2=0): .312,
                            fdict(T1=0, T2=1): .188,
                            fdict(T1=1, T2=0): .228,
                            fdict(T1=1, T2=1): .272})
    p_T1_T2___check = p_T1_T2.allclose(p_T1_T2___answer)

    p_T1_p_T2 = p_T1 * p_T2
    p_T1_p_T2.pprint()

    p_T1_T2_on_C_equal_n = (p_C * p_T1_on_C * p_T2_on_C).cond(C='n').norm()
    p_T1_T2_on_C_equal_n.pprint()
    p_T1_T2_on_C_equal_n___answer = PMF(dict.fromkeys(('C', 'T1', 'T2')),
                                        {fdict(T1=0, T2=0): .3,
                                         fdict(T1=0, T2=1): .2,
                                         fdict(T1=1, T2=0): .3,
                                         fdict(T1=1, T2=1): .2},
                                        cond=dict(C='n'))
    p_T1_T2_on_C_equal_n___check = p_T1_T2_on_C_equal_n.allclose(p_T1_T2_on_C_equal_n___answer)

    p_T1_on_C_equal_n = p_T1_on_C.at(C='n')
    p_T1_on_C_equal_n.pprint()
    p_T2_on_C_equal_n = p_T2_on_C.at(C='n')
    p_on_T1_T2_and_D_equal_1 = p_D_on_T1_T2.cond(D=1)
    p_T1_T2_on_C_equal_n_and_D_equal_1 = (p_T1_on_C_equal_n * p_T2_on_C_equal_n * p_on_T1_T2_and_D_equal_1).norm()
    p_T1_T2_on_C_equal_n_and_D_equal_1.pprint()

    p_T1_T2_on_C_equal_n_and_D_equal_1___alternative =\
        (p_C * p_T1_on_C * p_T2_on_C * p_D_on_T1_T2).cond(C='n', D=1).norm()
    p_T1_T2_on_C_equal_n_and_D_equal_1___alternative.pprint()

    p_T1_T2_on_C_equal_n_and_D_equal_1___answer = PMF(dict.fromkeys(('C', 'T1', 'T2', 'D')),
                                                      {fdict(T1=0, T2=0): .0,
                                                       fdict(T1=0, T2=1): .0,
                                                       fdict(T1=1, T2=0): .6,
                                                       fdict(T1=1, T2=1): .4},
                                                      cond=dict(C='n', D=1))
    p_T1_T2_on_C_equal_n_and_D_equal_1___answer.pprint()
    p_T1_T2_on_C_equal_n_and_D_equal_1___check =\
        p_T1_T2_on_C_equal_n_and_D_equal_1.allclose(p_T1_T2_on_C_equal_n_and_D_equal_1___alternative,
                                                    p_T1_T2_on_C_equal_n_and_D_equal_1___answer)

    assert p_T1___check & p_T2___check & p_T1_T2___check & p_T1_T2_on_C_equal_n___check &\
        p_T1_T2_on_C_equal_n_and_D_equal_1___check
