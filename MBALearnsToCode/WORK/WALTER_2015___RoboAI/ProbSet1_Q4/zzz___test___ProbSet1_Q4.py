from __future__ import print_function
from frozendict import frozendict as fdict
from pprint import pprint
from ProbabPy import DiscreteFinitePMF as PMF
from ProbabPyReason.HiddenMarkovModel import HiddenMarkovModel as HMM


def test___WALTER_2015___RoboAI___ProbSet1_Q4():
    """test: WALTER (2015) "Planning, Learning & Estimation in Robotics & A.I.": Problem Set 1, Question 4"""

    # Number of Time Periods
    T = 3

    state_prior = PMF(dict.fromkeys((('X', 0),)),
                      {fdict({('X', 0): 0}): .5,
                       fdict({('X', 0): 1}): .5})
    state_transition_template = PMF(dict.fromkeys((('X', -1), ('X', 0))),
                                    {fdict({('X', -1): 0, ('X', 0): 0}): .6,
                                     fdict({('X', -1): 0, ('X', 0): 1}): .4,
                                     fdict({('X', -1): 1, ('X', 0): 0}): .3,
                                     fdict({('X', -1): 1, ('X', 0): 1}): .7},
                                    cond={('X', -1): None})
    observation_template = PMF(dict.fromkeys((('X', 0), ('Z', 0))),
                               {fdict({('X', 0): 0, ('Z', 0): 0}): .8,
                                fdict({('X', 0): 0, ('Z', 0): 1}): .2,
                                fdict({('X', 0): 1, ('Z', 0): 0}): .2,
                                fdict({('X', 0): 1, ('Z', 0): 1}): .8},
                               cond={('X', 0): None})
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

    alpha_0 = PMF(dict.fromkeys((('X', 0), ('Z', 0))),
                  {fdict({('X', 0): 0, ('Z', 0): 0}): .4,
                   fdict({('X', 0): 1, ('Z', 0): 0}): .1},
                  scope={('Z', 0): 0})
    forward_0___check = forward[0].allclose(alpha_0)

    alpha_1 = PMF(dict.fromkeys((('X', 1), ('Z', 0), ('Z', 1))),
                  {fdict({('X', 1): 0, ('Z', 0): 0, ('Z', 1): 0}): .216,
                   fdict({('X', 1): 1, ('Z', 0): 0, ('Z', 1): 0}): .046},
                  scope={('Z', 0): 0, ('Z', 1): 0})
    forward_1___check = forward[1].allclose(alpha_1)

    alpha_2 = PMF(dict.fromkeys((('X', 2), ('Z', 0), ('Z', 1), ('Z', 2))),
                  {fdict({('X', 2): 0, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1}): .0287,
                   fdict({('X', 2): 1, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1}): .0949},
                  scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1})
    forward_2___check = forward[2].allclose(alpha_2, atol=1e-3)

    alpha_3 = PMF(dict.fromkeys((('X', 3), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  {fdict({('X', 3): 0, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): .0365,
                   fdict({('X', 3): 1, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): .0156},
                  scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0})
    forward_3___check = forward[3].allclose(alpha_3, atol=1e-2)

    print('"Backward" Probabilities:\n')
    backward = hmm.backward_factor(range(T + 1), all_z)

    for t in reversed(range(T + 1)):   # Recursively compute Backward factors
        print('Prob(z_%i to %i | X_%i) =' % (t + 1, T, t))
        backward[t].pprint()

    beta_3 = PMF(dict.fromkeys((('X', 3),)),
                 {fdict({('X', 3): 0}): 1.,
                  fdict({('X', 3): 1}): 1.},
                 cond={('X', 3): None})
    backward_3___check = backward[3].allclose(beta_3)

    beta_2 = PMF(dict.fromkeys((('X', 2), ('Z', 3))),
                 {fdict({('X', 2): 0, ('Z', 3): 0}): .56,
                  fdict({('X', 2): 1, ('Z', 3): 0}): .38},
                 cond={('X', 2): None},
                 scope={('Z', 3): 0})
    backward_2___check = backward[2].allclose(beta_2)

    beta_1 = PMF(dict.fromkeys((('X', 1), ('Z', 2), ('Z', 3))),
                 {fdict({('X', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): .1888,
                  fdict({('X', 1): 1, ('Z', 2): 1, ('Z', 3): 0}): .2464},
                 cond={('X', 1): None},
                 scope={('Z', 2): 1, ('Z', 3): 0})
    backward_1___check = backward[1].allclose(beta_1)

    beta_0 = PMF(dict.fromkeys((('X', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                 {fdict({('X', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): .1103,
                  fdict({('X', 0): 1, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): .0798},
                 cond={('X', 0): None},
                 scope={('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0})
    backward_0___check = backward[0].allclose(beta_0, atol=1e-3)

    print('Probability of each X conditional on all z values:\n')
    infer_state = hmm.infer_state(range(T + 1), all_z)
    for t in range(4):
        print('Prob(X_%i | all z) =' % t)
        infer_state[t].pprint()

    infer_0 = PMF(dict.fromkeys((('X', 0), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  {fdict({('X', 0): 0}): .4 * .1103,
                   fdict({('X', 0): 1}): .1 * .0798},
                  cond={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).norm()
    infer_0.pprint()
    infer_0___check = infer_state[0].allclose(infer_0, atol=1e-3)

    infer_1 = PMF(dict.fromkeys((('X', 1), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  {fdict({('X', 1): 0}): .216 * .1888,
                   fdict({('X', 1): 1}): .046 * .2464},
                  cond={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).norm()
    infer_1.pprint()
    infer_1___check = infer_state[1].allclose(infer_1)

    infer_2 = PMF(dict.fromkeys((('X', 2), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  {fdict({('X', 2): 0}): .0287 * .56,
                   fdict({('X', 2): 1}): .0949 * .38},
                  cond={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).norm()
    infer_2.pprint()
    infer_2___check = infer_state[2].allclose(infer_2, atol=1e-3)

    infer_3 = PMF(dict.fromkeys((('X', 3), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  {fdict({('X', 3): 0}): .0365,
                   fdict({('X', 3): 1}): .0156},
                  cond={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).norm()
    infer_3.pprint()
    infer_3___check = infer_state[3].allclose(infer_3, atol=1e-2)

    print('MAP of each X:\n')
    for t in range(T + 1):
        print('MAP Prob(X_%i | all z) =' % t)
        infer_state[t].max().pprint()

    # PART (B): VITERBI ALGORITHM
    print('\nVITERBI ALGORITHM:\n')

    all_z = [[0]]
    all_z += [all_z[0] + [0]]
    all_z += [all_z[1] + [1]]
    all_z += [all_z[2] + [0]]
    map_joint_distributions = []
    for t in range(T + 1):
        print('Most Likely Joint Distribution with actual z values up to z_%i:' % t)
        map_joint_distributions += [hmm.max_a_posteriori_joint_distributions(all_z[t])]
        map_joint_distributions[t].pprint()

    map_0 = PMF(dict.fromkeys((('X', 0), ('Z', 0))),
                {fdict({('X', 0): 0, ('Z', 0): 0}): .4},
                scope={('Z', 0): 0})
    map_0___check = map_joint_distributions[0].allclose(map_0)

    map_1 = PMF(dict.fromkeys((('X', 0), ('Z', 0), ('X', 1), ('Z', 1))),
                {fdict({('X', 0): 0, ('Z', 0): 0, ('X', 1): 0, ('Z', 1): 0}): .192},
                scope={('Z', 0): 0, ('Z', 1): 0})
    map_1___check = map_joint_distributions[1].allclose(map_1)

    map_2 = PMF(dict.fromkeys((('X', 0), ('Z', 0), ('X', 1), ('Z', 1), ('X', 2), ('Z', 2))),
                {fdict({('X', 0): 0, ('Z', 0): 0, ('X', 1): 0, ('Z', 1): 0, ('X', 2): 1, ('Z', 2): 1}): .0614},
                scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1})
    map_2___check = map_joint_distributions[2].allclose(map_2, atol=1e-3)

    map_3 = PMF(dict.fromkeys((('X', 0), ('Z', 0), ('X', 1), ('Z', 1), ('X', 2), ('Z', 2), ('X', 3), ('Z', 3))),
                {fdict({('X', 0): 0, ('Z', 0): 0, ('X', 1): 0, ('Z', 1): 0, ('X', 2): 1, ('Z', 2): 1,
                                      ('X', 3): 0, ('Z', 3): 0}): .0147},
                scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0})
    map_3___check = map_joint_distributions[3].allclose(map_3, atol=1e-2)

    map_state_sequence = hmm.max_a_posteriori_state_sequence(all_z[3])
    pprint(map_state_sequence)
    map_state_sequence___answer = [0, 0, 1, 0]
    map_state_sequence___check = map_state_sequence == map_state_sequence___answer

    assert forward_0___check & forward_1___check & forward_2___check & forward_3___check &\
        backward_0___check & backward_1___check & backward_2___check & backward_3___check &\
        infer_0___check & infer_1___check & infer_2___check & infer_3___check &\
        map_0___check & map_1___check & map_2___check & map_3___check & map_state_sequence___check
