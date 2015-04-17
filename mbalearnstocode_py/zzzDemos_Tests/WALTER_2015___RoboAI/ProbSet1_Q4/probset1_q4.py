# ROBOTICS & AI: ASSIGNMENT 01, QUESTION 04

from MBALearnsToCode_Py.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode_Py.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor


# Number of Time Periods
T = 3


# Set up Probability Factors / Joint Probability Distributions
print('\nPROBLEM SETUPS:\n')
prob = {}

print('Prob(X_0) =')
prob[('X', 0)] = Factor(DFDF({fdict({('X', 0): 0}): 0.5,
                              fdict({('X', 0): 1}): 0.5}))
prob[('X', 0)].print()
print('\n')

for t in range(T):
    print('Prob(X_%i | X_%i) =' % (t + 1, t))
    prob[(('X', t + 1), '|', ('X', t))] = Factor(DFDF({fdict({('X', t): 0, ('X', t + 1): 0}): 0.6,
                                                        fdict({('X', t): 0, ('X', t + 1): 1}): 0.4,
                                                        fdict({('X', t): 1, ('X', t + 1): 0}): 0.3,
                                                        fdict({('X', t): 1, ('X', t + 1): 1}): 0.7}),
                                                  conditions={('X', t): None})
    prob[(('X', t + 1), '|', ('X', t))].print()
    print('\n')

for t in range(T + 1):
    print('Prob(Z_%i | X%i) =' % (t, t))
    prob[(('Z', t), '|', ('X', t))] = Factor(DFDF({fdict({('X', t): 0, ('Z', t): 0}): 0.8,
                                                    fdict({('X', t): 0, ('Z', t): 1}): 0.2,
                                                    fdict({('X', t): 1, ('Z', t): 0}): 0.2,
                                                    fdict({('X', t): 1, ('Z', t): 1}): 0.8}),
                                              conditions={('X', t): None})
    prob[(('Z', t), '|', ('X', t))].print()
    print('\n')
    

all_z = {('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}
print('actual z values =\n', all_z)


# PART (A): FORWARD-BACKWARD ALGORITHM
print('\nFORWARD-BACKWARD ALGORITHM:\n')


print('"Forward" Probabilities:\n')
forward = dict()

t = 0
print('Prob(X_%i, z_%i) =' % (t, t))
forward[t] = prob[('X', t)].multiply(
    prob[(('Z', t), '|', ('X', t))].subs({('Z', t): all_z[('Z', t)]}))
forward[t].print()
print('\n')

for t in range(1, T + 1):   # Recursively compute Forward factors
    print('Prob(X_%i, z up to z_%i) =' % (t, t))
    forward[t] = forward[t - 1]\
        .multiply(prob[(('X', t), '|', ('X', t - 1))])\
        .eliminate(((('X', t - 1), 'sum', (0, 1)),))\
        .multiply(prob[(('Z', t), '|', ('X', t))].subs({('Z', t): all_z[('Z', t)]}))
    forward[t].print()
    print('\n')


print('"Backward" Probabilities:\n')
backward = dict()

print('Prob( | X_%i) = ' % T)
backward[T] = Factor(DFDF({fdict({('X', T): 0}): 1,
                           fdict({('X', T): 1}): 1}),
                     conditions={('X', T): None})
backward[T].print()
print('\n')

for t in reversed(range(T)):   # Recursively compute Backward factors
    print('Prob(z_%i to %i | X_%i) =' % (t + 1, T, t))
    backward[t] = prob[(('X', t + 1), '|', ('X', t))]\
        .multiply(prob[(('Z', t + 1), '|', ('X', t + 1))].subs({('Z', t + 1): all_z[('Z', t + 1)]}))\
        .multiply(backward[t + 1])\
        .eliminate(((('X', t + 1), 'sum', (0, 1)),))
    backward[t].print()
    print('\n')


print('Probability of all z values:\n')
print('Prob(all z values) =')
prob['all Z'] = forward[T].eliminate(((('X', T), 'sum', (0, 1)),))
prob['all Z'].print()
print('\n')


print('Probability of each X conditional on all z values:\n')
for t in range(4):
    prob[(('X', t), '|', 'all z')] = forward[t].multiply(backward[t]).condition(None, all_z).normalize()
    print('Prob(X_%i | all z) =' % t)
    prob[(('X', t), '|', 'all z')].print()
    print('\n')


# PART (B): VITERBI ALGORITHM
print('\nVITERBI ALGORITH:\n')

viterbi = {}

print('Most Likely Joint Distribution at t = 0 with actual z_0:')
viterbi[0] = forward[0].max()
viterbi[0].print()
print('\n')

for t in range(1, 4):
    print('Most Likely Joint Distribution up to t = %i with actual z values up to z_%i:' % (t, t))
    viterbi[t] = (viterbi[t - 1]\
        .multiply(prob[(('X', t), '|', ('X', t - 1))])\
        .multiply(prob[(('Z', t), '|', ('X', t))].subs({('Z', t): all_z[('Z', t)]}))).max()
    viterbi[t].print()
    print('\n')