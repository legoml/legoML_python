# ROBOTICS and AI HOMEWORK 01, QUESTION 02

from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from MBALearnsToCode.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor


alpha = 0.6

f_S0_S1 = Factor(DFDF({fdict(S0=0, S1=0): 0.9,
                       fdict(S0=0, S1=1): 0.1,
                       fdict(S0=1, S1=0): 0.1,
                       fdict(S0=1, S1=1): 0.9}))
f_S1_S2 = Factor(DFDF({fdict(S1=0, S2=0): alpha,
                       fdict(S1=0, S2=1): 1.0 - alpha,
                       fdict(S1=1, S2=0): 1.0,
                       fdict(S1=1, S2=1): alpha}))
f_S2_S3 = Factor(DFDF({fdict(S2=0, S3=0): alpha,
                       fdict(S2=0, S3=1): 1.0 - alpha,
                       fdict(S2=1, S3=0): 1.0,
                       fdict(S2=1, S3=1): alpha}))
f_S3_S4 = Factor(DFDF({fdict(S3=0, S4=0): 0.9,
                       fdict(S3=0, S4=1): 0.1,
                       fdict(S3=1, S4=0): 0.1,
                       fdict(S3=1, S4=1): 0.9}))

f_S1_on_S0_equal_1 = f_S0_S1.condition(None, dict(S0=1))
f_S3_on_S4_equal_0 = f_S3_S4.condition(None, dict(S4=0))
f_S1_on_S0_equal_1_and_S4_equal_0 = (f_S1_on_S0_equal_1.multiply(f_S1_S2, f_S2_S3, f_S3_on_S4_equal_0)
                                     .eliminate((('S2', 'sum', (0, 1)), ('S3', 'sum', (0, 1))))).normalize()
f_S1_on_S0_equal_1_and_S4_equal_0.pprint()

l0 = lambda a: 0.09 * a ** 2 + 0.02 * a * (1 - a) + 0.09 * (1 - a)
l1 = lambda a: 0.09 * a ** 2 + 1.62 * a + 0.09 * (1 - a)
p0 = l0(alpha)
p1 = l1(alpha)
s = p0 + p1
p0 /= s
p1 /= s
print(p0, p1)