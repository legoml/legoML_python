from pprint import pprint

unit_tests = dict()

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

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q1.MidTerm_Q1 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q1
unit_tests['WALTER_2015___RoboAI___MidTerm_Q1'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q1()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q2.MidTerm_Q2 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q2
unit_tests['WALTER_2015___RoboAI___MidTerm_Q2'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q2()

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.MidTerm_Q3.MidTerm_Q3 import\
    UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q3
unit_tests['WALTER_2015___RoboAI___MidTerm_Q3'] = UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q3()

pprint(unit_tests)