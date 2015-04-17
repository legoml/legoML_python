# ROBOTICS and AI HOMEWORK 01, QUESTION 01

from frozen_dict import FrozenDict as fdict
from MBALearnsToCode_Py.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from MBALearnsToCode_Py.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor


f_C = Factor(DFDF({fdict(C='a'): 0.3,
                   fdict(C='n'): 0.5,
                   fdict(C='l'): 0.2}))
f_T1_on_C = Factor(DFDF({fdict(C='a', T1=0): 0.3,
                         fdict(C='a', T1=1): 0.7,
                         fdict(C='n', T1=0): 0.5,
                         fdict(C='n', T1=1): 0.5,
                         fdict(C='l', T1=0): 0.8,
                         fdict(C='l', T1=1): 0.2}),
                   conditions=dict(C=None))
f_T2_on_C = Factor(DFDF({fdict(C='a', T2=0): 0.2,
                         fdict(C='a', T2=1): 0.8,
                         fdict(C='n', T2=0): 0.6,
                         fdict(C='n', T2=1): 0.4,
                         fdict(C='l', T2=0): 0.9,
                         fdict(C='l', T2=1): 0.1}),
                   conditions=dict(C=None))
f_D_on_T1_T2 = Factor(DFDF({fdict(T1=0, T2=0, D=0): 1.0,
                            fdict(T1=0, T2=0, D=1): 0.0,
                            fdict(T1=0, T2=1, D=0): 1.0,
                            fdict(T1=0, T2=1, D=1): 0.0,
                            fdict(T1=1, T2=0, D=0): 0.0,
                            fdict(T1=1, T2=0, D=1): 1.0,
                            fdict(T1=1, T2=1, D=0): 0.0,
                            fdict(T1=1, T2=1, D=1): 1.0}),
                      conditions=dict(T1=None, T2=None))
f_T1 = (f_C.multiply(f_T1_on_C)).eliminate((('C', 'sum', ('a', 'n', 'l')),)).normalize()
f_T1.print()
f_T2 = (f_C.multiply(f_T2_on_C)).eliminate((('C', 'sum', ('a', 'n', 'l')),)).normalize()
f_T2.print()
f_T1_T2 = (f_C.multiply(f_T1_on_C, f_T2_on_C)).eliminate((('C', 'sum', ('a', 'n', 'l')),)).normalize()
f_T1_T2.print()
f_T1_f_T2 = f_T1.multiply(f_T2).normalize()
f_T1_f_T2.print()
f_T1_T2_on_C_equal_n = f_C.multiply(f_T1_on_C, f_T2_on_C).condition(None, dict(C='n')).normalize()
f_T1_T2_on_C_equal_n.print()

f_T1_on_C_equal_n = f_T1_on_C.condition(None, dict(C='n'))
f_T2_on_C_equal_n = f_T2_on_C.condition(None, dict(C='n'))
f_T1_T2_on_D_equal_1 = f_D_on_T1_T2.condition(None, dict(D=1))
f_T1_T2_on_C_equal_n_and_D_equal_1 = (f_T1_on_C_equal_n.multiply(f_T2_on_C_equal_n,
                                                                 f_T1_T2_on_D_equal_1)).normalize()
f_T1_T2_on_C_equal_n_and_D_equal_1.print()
f_T1_T2_on_C_equal_n_and_D_equal_1___alternative =\
    (f_C.multiply(f_T1_on_C, f_T2_on_C, f_D_on_T1_T2).condition(None, dict(C='n', D=1))).normalize()
f_T1_T2_on_C_equal_n_and_D_equal_1___alternative.print()