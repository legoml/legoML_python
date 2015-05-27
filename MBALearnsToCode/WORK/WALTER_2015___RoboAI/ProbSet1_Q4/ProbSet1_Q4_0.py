from __future__ import print_function
from pprint import pprint
from sympy import log
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import discrete_finite_mass_function as pmf,\
    discrete_finite_mass_functions_all_close as pmf_allclose
from MBALearnsToCode.Classes.CLASSES___HiddenMarkovModels import HiddenMarkovModel as HMM


def UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q4():

    # Number of Time Periods
    T = 3

    state_prior = pmf(dict.fromkeys((('X', 0),)),
                      dict(mappings={fdict({('X', 0): 0}): -log(.5),
                                     fdict({('X', 0): 1}): -log(.5)}))
    state_transition_template = pmf(dict.fromkeys((('X', -1), ('X', 0))),
                                    dict(mappings={fdict({('X', -1): 0, ('X', 0): 0}): -log(.6),
                                                   fdict({('X', -1): 0, ('X', 0): 1}): -log(.4),
                                                   fdict({('X', -1): 1, ('X', 0): 0}): -log(.3),
                                                   fdict({('X', -1): 1, ('X', 0): 1}): -log(.7)}),
                                    conditions={('X', -1): None})
    observation_template = pmf(dict.fromkeys((('X', 0), ('Z', 0))),
                               dict(mappings={fdict({('X', 0): 0, ('Z', 0): 0}): -log(.8),
                                              fdict({('X', 0): 0, ('Z', 0): 1}): -log(.2),
                                              fdict({('X', 0): 1, ('Z', 0): 0}): -log(.2),
                                              fdict({('X', 0): 1, ('Z', 0): 1}): -log(.8)}),
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

    alpha_0 = pmf(dict.fromkeys((('X', 0), ('Z', 0))),
                  dict(mappings={fdict({('X', 0): 0, ('Z', 0): 0}): -log(.4),
                                 fdict({('X', 0): 1, ('Z', 0): 0}): -log(.1)}),
                  scope={('Z', 0): 0})
    forward_0___check = pmf_allclose(forward[0], alpha_0)

    alpha_1 = pmf(dict.fromkeys((('X', 1), ('Z', 0), ('Z', 1))),
                  dict(mappings={fdict({('X', 1): 0, ('Z', 0): 0, ('Z', 1): 0}): -log(.216),
                                 fdict({('X', 1): 1, ('Z', 0): 0, ('Z', 1): 0}): -log(.046)}),
                  scope={('Z', 0): 0, ('Z', 1): 0})
    forward_1___check = pmf_allclose(forward[1], alpha_1)

    alpha_2 = pmf(dict.fromkeys((('X', 2), ('Z', 0), ('Z', 1), ('Z', 2))),
                  dict(mappings={fdict({('X', 2): 0, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1}): -log(.0287),
                                 fdict({('X', 2): 1, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1}): -log(.0949)}),
                  scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1})
    forward_2___check = pmf_allclose(forward[2], alpha_2, atol=1e-3)

    alpha_3 = pmf(dict.fromkeys((('X', 3), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  dict(mappings={fdict({('X', 3): 0, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}):
                                     -log(.0365),
                                 fdict({('X', 3): 1, ('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}):
                                     -log(.0156)}),
                  scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0})
    forward_3___check = pmf_allclose(forward[3], alpha_3, atol=1e-2)

    print('"Backward" Probabilities:\n')
    backward = hmm.backward_factor(range(T + 1), all_z)

    for t in reversed(range(T + 1)):   # Recursively compute Backward factors
        print('Prob(z_%i to %i | X_%i) =' % (t + 1, T, t))
        backward[t].pprint()

    beta_3 = pmf(dict.fromkeys((('X', 3),)),
                 dict(mappings={fdict({('X', 3): 0}): -log(1.),
                                fdict({('X', 3): 1}): -log(1.)}),
                 conditions={('X', 3): None})
    backward_3___check = pmf_allclose(backward[3], beta_3)

    beta_2 = pmf(dict.fromkeys((('X', 2), ('Z', 3))),
                 dict(mappings={fdict({('X', 2): 0, ('Z', 3): 0}): -log(.56),
                                fdict({('X', 2): 1, ('Z', 3): 0}): -log(.38)}),
                 conditions={('X', 2): None},
                 scope={('Z', 3): 0})
    backward_2___check = pmf_allclose(backward[2], beta_2)

    beta_1 = pmf(dict.fromkeys((('X', 1), ('Z', 2), ('Z', 3))),
                 dict(mappings={fdict({('X', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): -log(.1888),
                                fdict({('X', 1): 1, ('Z', 2): 1, ('Z', 3): 0}): -log(.2464)}),
                 conditions={('X', 1): None},
                 scope={('Z', 2): 1, ('Z', 3): 0})
    backward_1___check = pmf_allclose(backward[1], beta_1)

    beta_0 = pmf(dict.fromkeys((('X', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                 dict(mappings={fdict({('X', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): -log(.1103),
                                fdict({('X', 0): 1, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}): -log(.0798)}),
                 conditions={('X', 0): None},
                 scope={('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0})
    backward_0___check = pmf_allclose(backward[0], beta_0, atol=1e-3)

    print('Probability of each X conditional on all z values:\n')
    infer_state = hmm.infer_state(range(T + 1), all_z)
    for t in range(4):
        print('Prob(X_%i | all z) =' % t)
        infer_state[t].pprint()

    infer_0 = pmf(dict.fromkeys((('X', 0), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  dict(mappings={fdict({('X', 0): 0}): -log(.4 * .1103),
                                 fdict({('X', 0): 1}): -log(.1 * .0798)}),
                  conditions={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).normalize()
    infer_0.pprint()
    infer_0___check = pmf_allclose(infer_state[0], infer_0, atol=1e-3)

    infer_1 = pmf(dict.fromkeys((('X', 1), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  dict(mappings={fdict({('X', 1): 0}): -log(.216 * .1888),
                                 fdict({('X', 1): 1}): -log(.046 * .2464)}),
                  conditions={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).normalize()
    infer_1.pprint()
    infer_1___check = pmf_allclose(infer_state[1], infer_1)

    infer_2 = pmf(dict.fromkeys((('X', 2), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  dict(mappings={fdict({('X', 2): 0}): -log(.0287 * .56),
                                 fdict({('X', 2): 1}): -log(.0949 * .38)}),
                  conditions={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).normalize()
    infer_2.pprint()
    infer_2___check = pmf_allclose(infer_state[2], infer_2, atol=1e-3)

    infer_3 = pmf(dict.fromkeys((('X', 3), ('Z', 0), ('Z', 1), ('Z', 2), ('Z', 3))),
                  dict(mappings={fdict({('X', 3): 0}): -log(.0365),
                                 fdict({('X', 3): 1}): -log(.0156)}),
                  conditions={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0}).normalize()
    infer_3.pprint()
    infer_3___check = pmf_allclose(infer_state[3], infer_3, atol=1e-2)

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

    map_0 = pmf(dict.fromkeys((('X', 0), ('Z', 0))),
                dict(mappings={fdict({('X', 0): 0, ('Z', 0): 0}): -log(.4)}),
                scope={('Z', 0): 0})
    map_0___check = pmf_allclose(map_joint_distributions[0], map_0)

    map_1 = pmf(dict.fromkeys((('X', 0), ('Z', 0), ('X', 1), ('Z', 1))),
                dict(mappings={fdict({('X', 0): 0, ('Z', 0): 0, ('X', 1): 0, ('Z', 1): 0}): -log(.192)}),
                scope={('Z', 0): 0, ('Z', 1): 0})
    map_1___check = pmf_allclose(map_joint_distributions[1], map_1)

    map_2 = pmf(dict.fromkeys((('X', 0), ('Z', 0), ('X', 1), ('Z', 1), ('X', 2), ('Z', 2))),
                dict(mappings={fdict({('X', 0): 0, ('Z', 0): 0, ('X', 1): 0, ('Z', 1): 0,
                                      ('X', 2): 1, ('Z', 2): 1}): -log(.0614)}),
                scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1})
    map_2___check = pmf_allclose(map_joint_distributions[2], map_2, atol=1e-3)

    map_3 = pmf(dict.fromkeys((('X', 0), ('Z', 0), ('X', 1), ('Z', 1), ('X', 2), ('Z', 2), ('X', 3), ('Z', 3))),
                dict(mappings={fdict({('X', 0): 0, ('Z', 0): 0, ('X', 1): 0, ('Z', 1): 0, ('X', 2): 1, ('Z', 2): 1,
                                      ('X', 3): 0, ('Z', 3): 0}): -log(.0147)}),
                scope={('Z', 0): 0, ('Z', 1): 0, ('Z', 2): 1, ('Z', 3): 0})
    map_3___check = pmf_allclose(map_joint_distributions[3], map_3, atol=1e-2)

    map_state_sequence = hmm.max_a_posteriori_state_sequence(all_z[3])
    pprint(map_state_sequence)
    map_state_sequence___answer = [0, 0, 1, 0]
    map_state_sequence___check = map_state_sequence == map_state_sequence___answer

    return forward_0___check & forward_1___check & forward_2___check & forward_3___check &\
        backward_0___check & backward_1___check & backward_2___check & backward_3___check &\
        infer_0___check & infer_1___check & infer_2___check & infer_3___check &\
        map_0___check & map_1___check & map_2___check & map_3___check & map_state_sequence___check