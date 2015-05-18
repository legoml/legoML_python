# ROBOTICS & AI: ASSIGNMENT 01, QUESTION 04

from pprint import pprint
from sympy import log
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import discrete_finite_mass_function as dfmf
from MBALearnsToCode.Classes import HiddenMarkovModel as HMM


# Number of Time Periods
T = 3

state_prior = dfmf(dict.fromkeys((('X', 0),)),
                   dict(mappings={fdict({('X', 0): 0}): -log(0.5),
                                  fdict({('X', 0): 1}): -log(0.5)}))
state_transition_template = dfmf(dict.fromkeys((('X', -1), ('X', 0))),
                                 dict(mappings={fdict({('X', -1): 0, ('X', 0): 0}): -log(0.6),
                                                fdict({('X', -1): 0, ('X', 0): 1}): -log(0.4),
                                                fdict({('X', -1): 1, ('X', 0): 0}): -log(0.3),
                                                fdict({('X', -1): 1, ('X', 0): 1}): -log(0.7)}),
                                 conditions={('X', -1): None})
observation_template = dfmf(dict.fromkeys((('X', 0), ('Z', 0))),
                            dict(mappings={fdict({('X', 0): 0, ('Z', 0): 0}): -log(0.8),
                                           fdict({('X', 0): 0, ('Z', 0): 1}): -log(0.2),
                                           fdict({('X', 0): 1, ('Z', 0): 0}): -log(0.2),
                                           fdict({('X', 0): 1, ('Z', 0): 1}): -log(0.8)}),
                            conditions={('X', 0): None})
hmm = HMM('X', 'Z', state_prior, state_transition_template, observation_template)


# Set up Probability Factors / Joint Probability Distributions
print('\nPROBLEM SETUPS:\n')

print('Prob(X_0) =')
hmm.state_prior_pdf.pprint()

for t in range(1, T + 1):
    print('Prob(X_%i | X_%i) =' % (t, t - 1))
    hmm.transition_pdf(t).pprint()

for t in range(T + 1):
    print('Prob(Z_%i | X_%i) =' % (t, t))
    hmm.observation_pdf(t).pprint()
    

all_z = {0: 0, 1: 0, 2: 1, 3: 0}
print('actual z values =\n', all_z)


# PART (A): FORWARD-BACKWARD ALGORITHM
print('\nFORWARD-BACKWARD ALGORITHM:\n')


print('"Forward" Probabilities:\n')
forward = hmm.forward_pdf(range(T + 1), all_z)

for t in range(T + 1):
    print('Prob(X_%i, z up to z_%i) =' % (t, t))
    forward[t].pprint()


print('"Backward" Probabilities:\n')
backward = hmm.backward_factor(range(T + 1), all_z)

for t in reversed(range(T + 1)):   # Recursively compute Backward factors
    print('Prob(z_%i to %i | X_%i) =' % (t + 1, T, t))
    backward[t].pprint()


print('Probability of each X conditional on all z values:\n')
infer_state = hmm.infer_state(range(T + 1), all_z)
for t in range(4):
    print('Prob(X_%i | all z) =' % t)
    infer_state[t].pprint()

print('MAP of each X:\n')
for t in range(4):
    print('MAP Prob(X_%i | all z) =' % t)
    infer_state[t].max().pprint()



# PART (B): VITERBI ALGORITHM
print('\nVITERBI ALGORITHM:\n')

print('Most Likely Joint Distribution with actual z values up to z_%i:' % T)
hmm.map_joint_distributions(all_z).pprint()
pprint(hmm.map_state_sequences(all_z))

all_y = [0, 0, 1, 0]
h = hmm.map_joint_distributions(all_y)
h.pprint()