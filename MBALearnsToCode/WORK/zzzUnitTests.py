from __future__ import print_function


unit_tests = dict()


from MBALearnsToCode.Functions.FUNCTIONS___Geometry2D import UNIT_TEST___Distances_and_Angles
unit_tests['Geometry2D___Distances_and_Angles'] = UNIT_TEST___Distances_and_Angles()

from MBALearnsToCode.WORK.KOLLER_2012___ProbGraphModels.Factors import\
    UNIT_TEST___KOLLER_2012___ProbGraphModels___Factors
unit_tests['KOLLER_2012___ProbGraphModels___Factors'] = UNIT_TEST___KOLLER_2012___ProbGraphModels___Factors()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q1.ProbSet1_Q1 import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q1
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q1'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q1()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q2.ProbSet1_Q2 import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q2
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q2'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q2(0.99) &\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q2(0.6)

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q4.ProbSet1_Q4 import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q4
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q4'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q4()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q5.RunViterbi import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q5
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q5'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q5()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet2_Q3.RunEKF import UNIT_TEST___WALTER_2015___RoboAI___ProbSet2_Q3
folder_path = '../MBALearnsToCode_Data/WALTER_2015___RoboAI/2.3 (EKF)/'
control_data_file_path = folder_path + 'U.txt'
measurement_data_file_path = folder_path + 'Z.txt'
ground_truth_file_path = folder_path + 'XYT.txt'
means_answer_file_path = folder_path + 'means.csv'
standard_deviations_answer_file_path = folder_path + 'stdevs.csv'
unit_tests['WALTER_2015___RoboAI___ProbSet2_Q3'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet2_Q3(
    control_data_file_path, measurement_data_file_path, ground_truth_file_path, means_answer_file_path,
    standard_deviations_answer_file_path)

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q1.MidTerm_Q1 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q1
unit_tests['WALTER_2015___RoboAI___MidTerm_Q1'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q1()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q2.MidTerm_Q2 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q2
unit_tests['WALTER_2015___RoboAI___MidTerm_Q2'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q2()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q3.MidTerm_Q3 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q3
unit_tests['WALTER_2015___RoboAI___MidTerm_Q3'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q3()


print(unit_tests)