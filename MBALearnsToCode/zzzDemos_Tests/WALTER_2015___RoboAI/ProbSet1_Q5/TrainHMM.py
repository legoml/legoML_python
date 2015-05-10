import sys
from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import discrete_finite_mass_function as dfmf
from MBALearnsToCode.zzzDemos_Tests.WALTER_2015___RoboAI.ProbSet1_Q5.DataSet import DataSet


def train_hmm(training_data_sequences):

    state_prior_counts = {}
    observation_counts = {}
    state_transition_counts = {}

    for i in range(len(training_data_sequences)):
        data_sequence = training_data_sequences[i]
        xy = [data_sequence.iloc[0]['xy']]
        ob = [data_sequence.iloc[0]['ob']]
        if fdict({'xy': xy[0]}) in state_prior_counts:
            state_prior_counts[fdict({'xy': xy[0]})] += 1
        else:
            state_prior_counts[fdict({'xy': xy[0]})] = 1
        if fdict({'xy': xy[0], 'ob': ob[0]}) in observation_counts:
            observation_counts[fdict({'xy': xy[0], 'ob': ob[0]})] += 1
        else:
            observation_counts[fdict({'xy': xy[0], 'ob': ob[0]})] = 1
        for t in range(1, len(data_sequence)):
            xy += [data_sequence.iloc[t]['xy']]
            ob += [data_sequence.iloc[t]['ob']]
            if fdict({'xy': xy[t - 1], 'next_xy': xy[t]}) in state_transition_counts:
                state_transition_counts[fdict({'xy': xy[t - 1], 'next_xy': xy[t]})] += 1
            else:
                state_transition_counts[fdict({'xy': xy[t - 1], 'next_xy': xy[t]})] = 1
            if fdict({'xy': xy[t], 'ob': ob[t]}) in observation_counts:
                observation_counts[fdict({'xy': xy[t], 'ob': ob[t]})] += 1
            else:
                observation_counts[fdict({'xy': xy[t], 'ob': ob[t]})] = 1

    state_prior_prob = Factor(DFDF(state_prior_counts)).normalize()
    state_transition_prob = Factor(DFDF(state_transition_counts), conditions={'xy': None}).normalize()
    observation_prob = Factor(DFDF(observation_counts), conditions={'xy': None}).normalize()

    return state_prior_prob, state_transition_prob, observation_prob


if __name__ == '__main__':
    file_name = sys.argv[1]
    print('\nReading %s...\n' %file_name)
    data_sequences = DataSet(file_name).data_sequences
    print('\nEstimating HMM Probabilities...\n')
    state_prior_prob, state_transition_prob, observation_prob = train_hmm(data_sequences)

    print('Estimated State Prior Probabilities:')
    state_prior_prob.print_factor()
    print('\n')

    print('Estimated State Transition Probabilities:')
    state_transition_prob.print_factor()
    print('\n')

    print('Estimated Observation Probabilities:')
    observation_prob.print_factor()
    print('\n')