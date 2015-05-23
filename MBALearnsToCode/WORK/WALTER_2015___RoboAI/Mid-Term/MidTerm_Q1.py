from sympy import log
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import\
    discrete_finite_mass_function as dfmf


p_C = dfmf(dict.fromkeys(('C',)),
           dict(mappings={fdict(C='a'): -log(0.3),
                          fdict(C='n'): -log(0.5),
                          fdict(C='l'): -log(0.2)}))
p_T1_on_C = dfmf(dict.fromkeys(('C', 'T1')),
                 dict(mappings={fdict(C='a', T1=0): -log(0.3),
                                fdict(C='a', T1=1): -log(0.7),
                                fdict(C='n', T1=0): -log(0.5),
                                fdict(C='n', T1=1): -log(0.5),
                                fdict(C='l', T1=0): -log(0.8),
                                fdict(C='l', T1=1): -log(0.2)}),
                 conditions=dict(C=None))
p_T2_on_C = dfmf(dict.fromkeys(('C', 'T2')),
                 dict(mappings={fdict(C='a', T2=0): -log(0.2),
                                fdict(C='a', T2=1): -log(0.8),
                                fdict(C='n', T2=0): -log(0.6),
                                fdict(C='n', T2=1): -log(0.4),
                                fdict(C='l', T2=0): -log(0.9),
                                fdict(C='l', T2=1): -log(0.1)}),
                 conditions=dict(C=None))
p_D_on_T1_T2 = dfmf(dict.fromkeys(('T1', 'T2', 'D')),
                    dict(mappings={fdict(T1=0, T2=0, D=0): -log(1.0),
                                   fdict(T1=0, T2=0, D=1): -log(0.0),
                                   fdict(T1=0, T2=1, D=0): -log(1.0),
                                   fdict(T1=0, T2=1, D=1): -log(0.0),
                                   fdict(T1=1, T2=0, D=0): -log(0.0),
                                   fdict(T1=1, T2=0, D=1): -log(1.0),
                                   fdict(T1=1, T2=1, D=0): -log(0.0),
                                   fdict(T1=1, T2=1, D=1): -log(1.0)}),
                    conditions=dict(T1=None, T2=None))
p_T1 = (p_C.multiply(p_T1_on_C)).marginalize(('C',))
p_T1.pprint()
p_T2 = (p_C.multiply(p_T2_on_C)).marginalize(('C',))
p_T2.pprint()
p_T1_T2 = (p_C.multiply(p_T1_on_C, p_T2_on_C)).marginalize(('C',))
p_T1_T2.pprint()
p_T1_p_T2 = p_T1.multiply(p_T2)
p_T1_p_T2.pprint()
p_T1_T2_on_C_equal_n = p_C.multiply(p_T1_on_C, p_T2_on_C).condition(dict(C='n')).normalize()
p_T1_T2_on_C_equal_n.pprint()

p_T1_on_C_equal_n = p_T1_on_C.condition(dict(C='n'))
p_T2_on_C_equal_n = p_T2_on_C.condition(dict(C='n'))
p_T1_T2_on_D_equal_1 = p_D_on_T1_T2.condition(dict(D=1))
p_T1_T2_on_C_equal_n_and_D_equal_1 = (p_T1_on_C_equal_n.multiply(p_T2_on_C_equal_n,
                                                                 p_T1_T2_on_D_equal_1)).normalize()
p_T1_T2_on_C_equal_n_and_D_equal_1.pprint()
p_T1_T2_on_C_equal_n_and_D_equal_1___alternative =\
    (p_C.multiply(p_T1_on_C, p_T2_on_C, p_D_on_T1_T2).condition(dict(C='n', D=1))).normalize()
p_T1_T2_on_C_equal_n_and_D_equal_1___alternative.pprint()