import sys
from sympy import log
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import discrete_finite_mass_function as dfmf
from MBALearnsToCode import DataSet


def train_hmm(training_data_sequences):

    state_prior_counts = {}
    observation_counts = {}
    transition_counts = {}

    for i in range(len(training_data_sequences)):
        data_sequence = training_data_sequences[i]
        xy = [data_sequence.iloc[0]['xy']]
        ob = [data_sequence.iloc[0]['ob']]
        if fdict({('xy', 0): xy[0]}) in state_prior_counts:
            state_prior_counts[fdict({('xy', 0): xy[0]})] += 1.
        else:
            state_prior_counts[fdict({('xy', 0): xy[0]})] = 1.
        if fdict({('xy', 0): xy[0], ('ob', 0): ob[0]}) in observation_counts:
            observation_counts[fdict({('xy', 0): xy[0], ('ob', 0): ob[0]})] += 1.
        else:
            observation_counts[fdict({('xy', 0): xy[0], ('ob', 0): ob[0]})] = 1.
        for t in range(1, len(data_sequence)):
            xy += [data_sequence.iloc[t]['xy']]
            ob += [data_sequence.iloc[t]['ob']]
            if fdict({('xy', -1): xy[t - 1], ('xy', 0): xy[t]}) in transition_counts:
                transition_counts[fdict({('xy', -1): xy[t - 1], ('xy', 0): xy[t]})] += 1.
            else:
                transition_counts[fdict({('xy', -1): xy[t - 1], ('xy', 0): xy[t]})] = 1.
            if fdict({('xy', 0): xy[t], ('ob', 0): ob[t]}) in observation_counts:
                observation_counts[fdict({('xy', 0): xy[t], ('ob', 0): ob[t]})] += 1.
            else:
                observation_counts[fdict({('xy', 0): xy[t], ('ob', 0): ob[t]})] = 1.

    state_prior_minus_log_counts = {k: -log(v) for k, v in state_prior_counts.items()}
    transition_minus_log_counts = {k: -log(v) for k, v in transition_counts.items()}
    observation_minus_log_counts = {k: -log(v) for k, v in observation_counts.items()}

    state_prior = dfmf(dict.fromkeys((('xy', 0), ('ob', 0))),
                       dict(mappings=state_prior_minus_log_counts)).normalize()
    transition_template = dfmf(dict.fromkeys((('xy', -1), ('xy', 0))),
                               dict(mappings=transition_minus_log_counts),
                               conditions={('xy', -1): None}).normalize()
    observation_template = dfmf(dict.fromkeys((('xy', 0), ('ob', 0))),
                                dict(mappings=observation_minus_log_counts),
                                conditions={('xy', 0): None}).normalize()

    return state_prior, transition_template, observation_template


if __name__ == '__main__':
    file_name = sys.argv[1]
    print('\nReading %s...\n' %file_name)
    data_sequences = DataSet(file_name).data_sequences
    print('\nEstimating HMM Probabilities...\n')
    state_prior, transition_template, observation_template = train_hmm(data_sequences)

    print('Estimated State Prior Probabilities:')
    state_prior.pprint()

    print('Estimated State Transition Probabilities:')
    transition_template.pprint()

    print('Estimated Observation Probabilities:')
    observation_template.pprint()