import sys
import pickle
from MBALearnsToCode.zzzDemos_Tests.WALTER_2015___RoboAI.ProbSet1_Q5.DataSet import DataSet
from MBALearnsToCode.zzzDemos_Tests.WALTER_2015___RoboAI.ProbSet1_Q5.TrainHMM import train_hmm
from MBALearnsToCode.UserDefinedClasses.CLASSES___HiddenMarkovModels import HMM


if __name__ == '__main__':

    training_data_file_name = sys.argv[1]
    testing_data_file_name = sys.argv[2]
    training_data_sequences = DataSet(training_data_file_name).data_sequences
    print('Estimating HMM Probabilities from Training Samples...')
    state_prior_prob, state_transition_prob, observation_prob = train_hmm(training_data_sequences)
    print('Estimating HMM Probabilities from Training Samples... done!')
    state_domain = ((1, 2), (1, 3), (1, 4),
                    (2, 1), (2, 3), (2, 4),
                    (3, 1), (3, 2), (3, 3),
                    (4, 2), (4, 3), (4, 4))
    hmm = HMM(('xy', 'next_xy'), 'ob', state_domain,
              state_prior_prob, state_transition_prob, observation_prob)

    m = {}
    testing_data_sequences = DataSet(testing_data_file_name).data_sequences
    if len(sys.argv) > 3:
        num_test_sequences_to_process = min(int(sys.argv[3]), len(testing_data_sequences))
    else:
        num_test_sequences_to_process = len(testing_data_sequences)
    for i in range(num_test_sequences_to_process):
        print('Estimating Most Likely State Sequence for Test Sample #%i' %i)
        testing_data_sequences[i]['most_likely_xy_sequence'] =\
            tuple(hmm.map_state_sequences(list(testing_data_sequences[i]['ob'])))[0]

    output_file = 'testing_data_sequences_with_MAP.PICKLE'
    print('Outputing to:', output_file)
    pickle.dump(testing_data_sequences, open(output_file, 'wb'))
    print('done!')