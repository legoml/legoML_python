from pprint import pprint

unit_tests = dict()

from MBALearnsToCode.WORK.KOLLER_2012___ProbGraphModels.Factors_0 import\
    UNIT_TEST___KOLLER_2012___ProbGraphModels___Factors
unit_tests['KOLLER_2012___ProbGraphModels___Factors'] = UNIT_TEST___KOLLER_2012___ProbGraphModels___Factors()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q1.ProbSet1_Q1_0 import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q1
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q1'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q1()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q2.ProbSet1_Q2_0 import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q2
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q2'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q2(0.99) &\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q2(0.6)

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q4.ProbSet1_Q4_0 import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q4
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q4'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q4()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet1_Q5.RunViterbi import\
    UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q5
unit_tests['WALTER_2015___RoboAI___ProbSet1_Q5'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet1_Q5()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.ProbSet2_Q3.RunEKF import UNIT_TEST___WALTER_2015___RoboAI___ProbSet2_Q3
folder_path = './MBALearnsToCode/WORK/WALTER_2015___RoboAI/ProbSet2_Q3/'
control_data_file_path = folder_path + 'U.txt'
measurement_data_file_path = folder_path + 'Z.txt'
ground_truth_file_path = folder_path + 'XYT.txt'
unit_tests['WALTER_2015___RoboAI___ProbSet2_Q3'] = UNIT_TEST___WALTER_2015___RoboAI___ProbSet2_Q3(
    control_data_file_path, measurement_data_file_path, ground_truth_file_path)

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q1.MidTerm_Q1_0 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q1
unit_tests['WALTER_2015___RoboAI___MidTerm_Q1'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q1()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q2.MidTerm_Q2_0 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q2
unit_tests['WALTER_2015___RoboAI___MidTerm_Q2'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q2()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q3.MidTerm_Q3_0 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q3
unit_tests['WALTER_2015___RoboAI___MidTerm_Q3'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q3()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.RoboSoccer. zzzUnitTestGradients import\
    UNIT_TEST___WALTER_2015___RoboAI___RoboSoccer___FunctionGradients
unit_tests['WALTER_2015___RoboAI___RoboSoccer___FunctionGradients'] =\
    UNIT_TEST___WALTER_2015___RoboAI___RoboSoccer___FunctionGradients()

pprint(unit_tests)