from __future__ import print_function
import pickle
import sys
from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q5.DataSet import DataSet
from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q5.TrainHMM import train_hmm
from MBALearnsToCode.Classes.CLASSES___HiddenMarkovModels import HiddenMarkovModel as HMM


def UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q5():
    folder_path = '../../MBALearnsToCode_Data/WALTER_2015___RoboAI/1.5 (HMM Random Walk)/'
    training_data_file_name = folder_path + 'randomwalk.train.txt'
    testing_data_file_name = folder_path + 'randomwalk.test.txt'
    training_data_sequences = DataSet(training_data_file_name).data_sequences
    print('Estimating HMM Probabilities from Training Samples...')
    state_prior, transition_template, observation_template = train_hmm(training_data_sequences)
    print('Estimating HMM Probabilities from Training Samples... done!')
    state_prior.pprint()
    transition_template.pprint()
    observation_template.pprint()
    hmm = HMM('xy', 'ob', state_prior, transition_template, observation_template)
    testing_data_sequences = DataSet(testing_data_file_name).data_sequences
    s = hmm.max_a_posteriori_state_sequence(list(testing_data_sequences[0]['ob']))


if __name__ == '__main__':

    training_data_file_name = sys.argv[1]
    testing_data_file_name = sys.argv[2]
    training_data_sequences = DataSet(training_data_file_name).data_sequences
    print('Estimating HMM Probabilities from Training Samples...')
    state_prior, transition_template, observation_template = train_hmm(training_data_sequences)
    print('Estimating HMM Probabilities from Training Samples... done!')
    hmm = HMM('xy', 'ob', state_prior, transition_template, observation_template)

    m = {}
    testing_data_sequences = DataSet(testing_data_file_name).data_sequences
    if len(sys.argv) > 3:
        num_test_sequences_to_process = min(int(sys.argv[3]), len(testing_data_sequences))
    else:
        num_test_sequences_to_process = len(testing_data_sequences)
    for i in range(num_test_sequences_to_process):
        print('Estimating Most Likely State Sequence for Test Sample #%i' %i)
        testing_data_sequences[i]['most_likely_xy_sequence'] =\
            hmm.max_a_posteriori_state_sequence(list(testing_data_sequences[i]['ob']))

    output_file = 'testing_data_sequences_with_MAP.PICKLE'
    print('Outputing to:', output_file)
    pickle.dump(testing_data_sequences, open(output_file, 'wb'))
    print('done!')
