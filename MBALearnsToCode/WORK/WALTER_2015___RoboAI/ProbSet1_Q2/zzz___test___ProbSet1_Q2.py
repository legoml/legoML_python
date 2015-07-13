from frozendict import frozendict as fdict
from ProbabPy import DiscreteFinitePMF as PMF


def test___WALTER_2015___RoboAI___ProbSet1_Q2(alpha=.6):   # Symbol('alpha')
    """test: WALTER (2015) "Planning, Learning & Estimation in Robotics & A.I.": Problem Set 1, Question 2"""
    p_S0_S1 = PMF(dict.fromkeys(('S0', 'S1')),
                  {fdict(S0=0, S1=0): .9,
                   fdict(S0=0, S1=1): .1,
                   fdict(S0=1, S1=0): .1,
                   fdict(S0=1, S1=1): .9})
    p_S1_S2 = PMF(dict.fromkeys(('S1', 'S2')),
                  {fdict(S1=0, S2=0): alpha,
                   fdict(S1=0, S2=1): 1. - alpha,
                   fdict(S1=1, S2=0): 1. - alpha,
                   fdict(S1=1, S2=1): alpha})
    p_S2_S3 = PMF(dict.fromkeys(('S2', 'S3')),
                  {fdict(S2=0, S3=0): alpha,
                   fdict(S2=0, S3=1): 1. - alpha,
                   fdict(S2=1, S3=0): 1. - alpha,
                   fdict(S2=1, S3=1): alpha})
    p_S3_S4 = PMF(dict.fromkeys(('S3', 'S4')),
                  {fdict(S3=0, S4=0): .9,
                   fdict(S3=0, S4=1): .1,
                   fdict(S3=1, S4=0): .1,
                   fdict(S3=1, S4=1): .9})

    p_S1_on_S0_equal_1 = p_S0_S1.cond(S0=1)
    p_S3_on_S4_equal_0 = p_S3_S4.cond(S4=0)
    p_S1_on_S0_equal_1_and_S4_equal_0 = (p_S1_on_S0_equal_1 * p_S1_S2 * p_S2_S3 * p_S3_on_S4_equal_0)\
                                         .marg('S2', 'S3').norm()
    p_S1_on_S0_equal_1_and_S4_equal_0.pprint()

    lambda_0 = lambda a: .09 * (2 * a ** 2 - 2 * a + 1) + .02 * a * (1. - a)
    lambda_1 = lambda a: .09 * (2 * a ** 2 - 2 * a + 1) + 1.62 * a * (1. - a)
    p_S1_on_S0_equal_1_and_S4_equal_0___answer = PMF(dict.fromkeys(('S0', 'S1', 'S4')),
                                                     {fdict(S1=0): lambda_0(alpha),
                                                      fdict(S1=1): lambda_1(alpha)},
                                                     cond=dict(S0=1, S4=0)).norm()

    assert p_S1_on_S0_equal_1_and_S4_equal_0.allclose(p_S1_on_S0_equal_1_and_S4_equal_0___answer)
