# ROBOTICS and AI HOMEWORK 01, QUESTION 02

from sympy import log, Symbol
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import\
    discrete_finite_mass_function as dfmf


alpha =  0.6#Symbol('alpha')   #, 0.99

p_S0_S1 = dfmf(dict.fromkeys(('S0', 'S1')),
               dict(mappings={fdict(S0=0, S1=0): -log(0.9),
                              fdict(S0=0, S1=1): -log(0.1),
                              fdict(S0=1, S1=0): -log(0.1),
                              fdict(S0=1, S1=1): -log(0.9)}))
p_S1_S2 = dfmf(dict.fromkeys(('S1', 'S2')),
               dict(mappings={fdict(S1=0, S2=0): -log(alpha),
                              fdict(S1=0, S2=1): -log(1.0 - alpha),
                              fdict(S1=1, S2=0): -log(1.0 - alpha),
                              fdict(S1=1, S2=1): -log(alpha)}))
p_S2_S3 = dfmf(dict.fromkeys(('S2', 'S3')),
               dict(mappings={fdict(S2=0, S3=0): -log(alpha),
                              fdict(S2=0, S3=1): -log(1.0 - alpha),
                              fdict(S2=1, S3=0): -log(1.0 - alpha),
                              fdict(S2=1, S3=1): -log(alpha)}))
p_S3_S4 = dfmf(dict.fromkeys(('S3', 'S4')),
               dict(mappings={fdict(S3=0, S4=0): -log(0.9),
                              fdict(S3=0, S4=1): -log(0.1),
                              fdict(S3=1, S4=0): -log(0.1),
                              fdict(S3=1, S4=1): -log(0.9)}))

p_S1_on_S0_equal_1 = p_S0_S1.condition(dict(S0=1))
p_S3_on_S4_equal_0 = p_S3_S4.condition(dict(S4=0))
p_S1_on_S0_equal_1_and_S4_equal_0 = (p_S1_on_S0_equal_1.multiply(p_S1_S2, p_S2_S3, p_S3_on_S4_equal_0)
                                     .marginalize(('S2', 'S3'))).normalize()
p_S1_on_S0_equal_1_and_S4_equal_0.pprint()