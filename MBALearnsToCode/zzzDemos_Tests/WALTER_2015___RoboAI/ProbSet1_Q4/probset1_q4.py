# ROBOTICS & AI: ASSIGNMENT 01, QUESTION 04

from pprint import pprint
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF
from MBALearnsToCode.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor
from MBALearnsToCode.UserDefinedClasses.CLASSES___HiddenMarkovModels import HMM


# Number of Time Periods
T = 3

state_prior = Factor(DFDF({fdict(X=0): 0.5,
                           fdict(X=1): 0.5}))
state_transition_likelihood = Factor(DFDF({fdict(X=0, next_X=0): 0.6,
                                           fdict(X=0, next_X=1): 0.4,
                                           fdict(X=1, next_X=0): 0.3,
                                           fdict(X=1, next_X=1): 0.7}),
                                     conditions=dict(X=None))
observation_likelihood = Factor(DFDF({fdict(X=0, Z=0): 0.8,
                                      fdict(X=0, Z=1): 0.2,
                                      fdict(X=1, Z=0): 0.2,
                                      fdict(X=1, Z=1): 0.8}),
                                conditions=dict(X=None))
hmm = HMM(('X', 'next_X'), 'Z', (0, 1), state_prior, state_transition_likelihood, observation_likelihood)


# Set up Probability Factors / Joint Probability Distributions
print('\nPROBLEM SETUPS:\n')

print('Prob(X_0) =')
hmm.state_prior_factor().print()
print('\n')

for t in range(1, T + 1):
    print('Prob(X_%i | X_%i) =' % (t, t - 1))
    hmm.state_transition_factor(t).print()
    print('\n')

for t in range(T + 1):
    print('Prob(Z_%i | X_%i) =' % (t, t))
    hmm.observation_factor(t).print()
    print('\n')
    

all_z = {0: 0, 1: 0, 2: 1, 3: 0}
print('actual z values =\n', all_z)


# PART (A): FORWARD-BACKWARD ALGORITHM
print('\nFORWARD-BACKWARD ALGORITHM:\n')


print('"Forward" Probabilities:\n')
forward = hmm.forward_factor(range(T + 1), all_z)

for t in range(T + 1):
    print('Prob(X_%i, z up to z_%i) =' % (t, t))
    forward[t].print()
    print('\n')


print('"Backward" Probabilities:\n')
backward = hmm.backward_factor(range(T + 1), all_z)

for t in reversed(range(T + 1)):   # Recursively compute Backward factors
    print('Prob(z_%i to %i | X_%i) =' % (t + 1, T, t))
    backward[t].print()
    print('\n')



print('Probability of each X conditional on all z values:\n')
infer_state = hmm.infer_state(range(T + 1), all_z)
for t in range(4):
    print('Prob(X_%i | all z) =' % t)
    infer_state[t].print()
    print('\n')


# PART (B): VITERBI ALGORITHM
print('\nVITERBI ALGORITHM:\n')

print('Most Likely Joint Distribution with actual z values up to z_%i:' % T)
pprint(hmm.map_state_sequences(all_z))