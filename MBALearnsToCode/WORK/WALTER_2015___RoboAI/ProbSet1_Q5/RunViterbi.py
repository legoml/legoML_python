import sys
import pickle
from MBALearnsToCode import DataSet
from MBALearnsToCode import train_hmm
from MBALearnsToCode.Classes import HiddenMarkovModel as HMM


training_data_file_name =\
    './MBALearnsToCode/zzzDemos_Tests/WALTER_2015___RoboAI/ProbSet1_Q5/randomwalk.train.txt'
testing_data_file_name =\
    './MBALearnsToCode/zzzDemos_Tests/WALTER_2015___RoboAI/ProbSet1_Q5/randomwalk.test.txt'
training_data_sequences = DataSet(training_data_file_name).data_sequences
print('Estimating HMM Probabilities from Training Samples...')
state_prior, transition_template, observation_template = train_hmm(training_data_sequences)
print('Estimating HMM Probabilities from Training Samples... done!')
state_prior.pprint()
transition_template.pprint()
observation_template.pprint()
hmm = HMM('xy', 'ob', state_prior, transition_template, observation_template)
testing_data_sequences = DataSet(testing_data_file_name).data_sequences
s = hmm.map_state_sequences(list(testing_data_sequences[0]['ob']))


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
            hmm.map_state_sequences(list(testing_data_sequences[i]['ob']))

    output_file = 'testing_data_sequences_with_MAP.PICKLE'
    print('Outputing to:', output_file)
    pickle.dump(testing_data_sequences, open(output_file, 'wb'))
    print('done!')