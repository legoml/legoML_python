from frozendict import frozendict as fdict
from ProbabPy import DiscreteFinitePMF as PMF


def test___KOLLER_2012___FactorOperations():
    """test: KOLLER (2012) "Probabilistic Graphical Models": Factors"""
    f = PMF(dict.fromkeys(('a', 'b')),
            {fdict(a=1, b=1): .5,
             fdict(a=1, b=2): .8,
             fdict(a=2, b=1): .1,
             fdict(a=2, b=2): .0,
             fdict(a=3, b=1): .3,
             fdict(a=3, b=2): .9})

    f_1 = PMF(dict.fromkeys(('b', 'c')),
              {fdict(b=1, c=1): .5,
               fdict(b=1, c=2): .7,
               fdict(b=2, c=1): .1,
               fdict(b=2, c=2): .2})
    f = f * f_1
    f.pprint()
    f___answer = PMF(dict.fromkeys(('a', 'b', 'c')),
                     {fdict(a=1, b=1, c=1): .25,
                      fdict(a=1, b=1, c=2): .35,
                      fdict(a=1, b=2, c=1): .08,
                      fdict(a=1, b=2, c=2): .16,
                      fdict(a=2, b=1, c=1): .05,
                      fdict(a=2, b=1, c=2): .07,
                      fdict(a=2, b=2, c=1): .0,
                      fdict(a=2, b=2, c=2): .0,
                      fdict(a=3, b=1, c=1): .15,
                      fdict(a=3, b=1, c=2): .21,
                      fdict(a=3, b=2, c=1): .09,
                      fdict(a=3, b=2, c=2): .18})
    f___check = f.allclose(f___answer)
    print('Factor Multiplication check:', f___check)

    m = f.marg('b')
    m.pprint()
    m___answer = PMF(dict.fromkeys(('a', 'c')),
                     {fdict(a=1, c=1): .33,
                      fdict(a=1, c=2): .51,
                      fdict(a=2, c=1): .05,
                      fdict(a=2, c=2): .07,
                      fdict(a=3, c=1): .24,
                      fdict(a=3, c=2): .39})
    m___check = m.allclose(m___answer)
    print('Factor Marginalization check:', m___check)

    assert f___check & m___check
