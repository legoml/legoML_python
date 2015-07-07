from __future__ import division, print_function
from frozendict import frozendict as fdict
from ProbabPy import DiscreteFinitePMF as PMF
from MBALearnsToCode.Classes.CLASSES___HiddenMarkovModels import HiddenMarkovModel as HMM


def test___WALTER_2015___RoboAI___MidTerm_Q1():
    """test: WALTER (2015) "Planning, Learning & Estimation in Robotics & A.I.": Mid-Term Exam, Question 1"""

    # Number of Time Periods
    T = 2

    state_prior = PMF(dict.fromkeys((('S', 0),)),
                      {fdict({('S', 0): 1}): 1 / 3,
                       fdict({('S', 0): 2}): 1 / 3,
                       fdict({('S', 0): 3}): 1 / 3})
    state_transition_template = PMF(dict.fromkeys((('S', -1), ('S', 0))),
                                    {fdict({('S', -1): 1, ('S', 0): 1}): .2,
                                     fdict({('S', -1): 1, ('S', 0): 2}): .3,
                                     fdict({('S', -1): 1, ('S', 0): 3}): .5,
                                     fdict({('S', -1): 2, ('S', 0): 1}): .2,
                                     fdict({('S', -1): 2, ('S', 0): 2}): .7,
                                     fdict({('S', -1): 2, ('S', 0): 3}): .1,
                                     fdict({('S', -1): 3, ('S', 0): 1}): .4,
                                     fdict({('S', -1): 3, ('S', 0): 2}): .2,
                                     fdict({('S', -1): 3, ('S', 0): 3}): .4},
                                    conditions={('S', -1): None})
    observation_template = PMF(dict.fromkeys((('S', 0), ('Z', 0))),
                               {fdict({('S', 0): 1, ('Z', 0): 1}): .6,
                                fdict({('S', 0): 1, ('Z', 0): 2}): .1,
                                fdict({('S', 0): 1, ('Z', 0): 3}): .3,
                                fdict({('S', 0): 2, ('Z', 0): 1}): .1,
                                fdict({('S', 0): 2, ('Z', 0): 2}): .7,
                                fdict({('S', 0): 2, ('Z', 0): 3}): .2,
                                fdict({('S', 0): 3, ('Z', 0): 1}): .2,
                                fdict({('S', 0): 3, ('Z', 0): 2}): .3,
                                fdict({('S', 0): 3, ('Z', 0): 3}): .5},
                               conditions={('S', 0): None})
    hmm = HMM('S', 'Z', state_prior, state_transition_template, observation_template)

    # Set up Probability Factors / Joint Probability Distributions
    print('\nPROBLEM SETUPS:\n')

    print('Prob(S_0) =')
    hmm.state_prior_pdf.pprint()

    for t in range(1, T + 1):
        print('Prob(S_%i | S_%i) =' % (t, t - 1))
        hmm.transition_pdf(t).pprint()

    for t in range(T + 1):
        print('Prob(Z_%i | S_%i) =' % (t, t))
        hmm.observation_pdf(t).pprint()

    all_z = {0: 2, 1: 3, 2: 1}
    print('actual z values =\n', all_z)

    print('\nFORWARD-BACKWARD ALGORITHM:\n')

    print('"Forward" Probabilities:\n')
    forward = hmm.forward_pdf(range(T + 1), all_z)

    for t in range(T + 1):
        print('Prob(S_%i, z up to z_%i) =' % (t, t))
        forward[t].pprint()

    alpha_0 = PMF(dict.fromkeys((('S', 0), ('Z', 0))),
                  {fdict({('S', 0): 1, ('Z', 0): 2}): .1 / 3,
                   fdict({('S', 0): 2, ('Z', 0): 2}): .7 / 3,
                   fdict({('S', 0): 3, ('Z', 0): 2}): .3 / 3},
                  scope={('Z', 0): 2})
    forward_0___check = forward[0].allclose(alpha_0)

    alpha_1 = PMF(dict.fromkeys((('S', 1), ('Z', 0), ('Z', 1))),
                  {fdict({('S', 1): 1, ('Z', 0): 2, ('Z', 1): 3}): .084 / 3,
                   fdict({('S', 1): 2, ('Z', 0): 2, ('Z', 1): 3}): .116 / 3,
                   fdict({('S', 1): 3, ('Z', 0): 2, ('Z', 1): 3}): .12 / 3},
                  scope={('Z', 0): 2, ('Z', 1): 3})
    forward_1___check = forward[1].allclose(alpha_1)

    alpha_2 = PMF(dict.fromkeys((('S', 2), ('Z', 0), ('Z', 1), ('Z', 2))),
                  {fdict({('S', 2): 1, ('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1}): .0528 / 3,
                   fdict({('S', 2): 2, ('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1}): .013 / 3,
                   fdict({('S', 2): 3, ('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1}): .0203 / 3},
                  scope={('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1})
    forward_2___check = forward[2].allclose(alpha_2, atol=1e-2)

    print('"Backward" Probabilities:\n')
    backward = hmm.backward_factor(range(T + 1), all_z)

    for t in reversed(range(T + 1)):   # Recursively compute Backward factors
        print('Prob(z_%i to %i | S_%i) =' % (t + 1, T, t))
        backward[t].pprint()

    beta_2 = PMF(dict.fromkeys((('S', 2),)),
                 {fdict({('S', 2): 1}): 1.,
                  fdict({('S', 2): 2}): 1.,
                  fdict({('S', 2): 3}): 1.},
                 conditions={('S', 2): None})
    backward_2___check = backward[2].allclose(beta_2)

    beta_1 = PMF(dict.fromkeys((('S', 1), ('Z', 2))),
                 {fdict({('S', 1): 1, ('Z', 2): 1}): .25,
                  fdict({('S', 1): 2, ('Z', 2): 1}): .21,
                  fdict({('S', 1): 3, ('Z', 2): 1}): .34},
                 conditions={('S', 1): None},
                 scope={('Z', 2): 1})
    backward_1___check = backward[1].allclose(beta_1)

    beta_0 = PMF(dict.fromkeys((('S', 0), ('Z', 1), ('Z', 2))),
                 {fdict({('S', 0): 1, ('Z', 1): 3, ('Z', 2): 1}): .1126,
                  fdict({('S', 0): 2, ('Z', 1): 3, ('Z', 2): 1}): .0614,
                  fdict({('S', 0): 3, ('Z', 1): 3, ('Z', 2): 1}): .1064},
                 conditions={('S', 0): None},
                 scope={('Z', 1): 3, ('Z', 2): 1})
    backward_0___check = backward[0].allclose(beta_0)

    print('Probability of each X conditional on all z values:\n')
    infer_state = hmm.infer_state(range(T + 1), all_z)
    for t in range(T + 1):
        print('Prob(S_%i | all z) =' % t)
        infer_state[t].pprint()

    infer_0 = PMF(dict.fromkeys((('S', 0), ('Z', 0), ('Z', 1), ('Z', 2))),
                  {fdict({('S', 0): 1}): .0038,
                   fdict({('S', 0): 2}): .0143,
                   fdict({('S', 0): 3}): .0106},
                  conditions={('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1}).norm()
    infer_0.pprint()
    infer_0___check = infer_state[0].allclose(infer_0, atol=1e-1)

    infer_1 = PMF(dict.fromkeys((('S', 1), ('Z', 0), ('Z', 1), ('Z', 2))),
                  {fdict({('S', 1): 1}): .007,
                   fdict({('S', 1): 2}): .0081,
                   fdict({('S', 1): 3}): .0136},
                  conditions={('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1}).norm()
    infer_1.pprint()
    infer_1___check = infer_state[1].allclose(infer_1, atol=1e-2)

    infer_2 = PMF(dict.fromkeys((('S', 2), ('Z', 0), ('Z', 1), ('Z', 2))),
                  {fdict({('S', 2): 1}): .0176,
                   fdict({('S', 2): 2}): .0043,
                   fdict({('S', 2): 3}): .0068},
                  conditions={('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1}).norm()
    infer_2.pprint()
    infer_2___check = infer_state[2].allclose(infer_2, atol=1e-1)

    print('MAP of each X:\n')
    for t in range(T + 1):
        print('MAP Prob(S_%i | all z) =' % t)
        infer_state[t].max().pprint()

    print('\nVITERBI ALGORITHM:\n')

    all_z = [[2]]
    all_z += [all_z[0] + [3]]
    all_z += [all_z[1] + [1]]
    map_joint_distributions = []
    for t in range(T + 1):
        print('Most Likely Joint Distribution with actual z values up to z_%i:' % t)
        map_joint_distributions += [hmm.max_a_posteriori_joint_distributions(all_z[t])]
        map_joint_distributions[t].pprint()

    map_0 = PMF(dict.fromkeys((('S', 0), ('Z', 0))),
                {fdict({('S', 0): 2, ('Z', 0): 2}): .7 / 3},
                scope={('Z', 0): 2})
    map_0___check = map_joint_distributions[0].allclose(map_0)

    map_1 = PMF(dict.fromkeys((('S', 0), ('Z', 0), ('S', 1), ('Z', 1))),
                 {fdict({('S', 0): 2, ('Z', 0): 2, ('S', 1): 2, ('Z', 1): 3}): .0327},
                 scope={('Z', 0): 2, ('Z', 1): 3})
    map_1___check = map_joint_distributions[1].allclose(map_1, atol=1e-3)

    map_2 = PMF(dict.fromkeys((('S', 0), ('Z', 0), ('S', 1), ('Z', 1), ('S', 2), ('Z', 2))),
                {fdict({('S', 0): 3, ('Z', 0): 2, ('S', 1): 3, ('Z', 1): 3, ('S', 2): 1, ('Z', 2): 1}): .0048},
                scope={('Z', 0): 2, ('Z', 1): 3, ('Z', 2): 1})
    map_2___check = map_joint_distributions[2].allclose(map_2)

    map_state_sequence = hmm.max_a_posteriori_state_sequence(all_z[2])
    print(map_state_sequence)
    map_state_sequence___answer = [3, 3, 1]
    map_state_sequence___check = map_state_sequence == map_state_sequence___answer

    assert forward_0___check & forward_1___check & forward_2___check &\
        backward_0___check & backward_1___check & backward_2___check &\
        infer_0___check & infer_1___check & infer_2___check &\
        map_0___check & map_1___check & map_2___check & map_state_sequence___check
