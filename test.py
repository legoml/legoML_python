


file_name = 'C:/Cloud/Dropbox/MBALearnsToCode/legoML_python/MBALearnsToCode/zzzDemos_Tests/WALTER_2015___RoboAI/ProbSet1_Q5/randomwalk.train.txt'
data_sequences = DataSet(file_name).data_sequences
state_prior_prob, state_transition_prob, observation_prob = train_hmm(data_sequences)

print('Estimated State Prior Probabilities:')
state_prior_prob.print()
print('\n')

print('Estimated State Transition Probabilities:')
state_transition_prob.print()
print('\n')

print('Estimated Observation Probabilities:')
observation_prob.print()
print('\n')