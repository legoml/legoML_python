from numpy import allclose, loadtxt, sqrt
from RunEKF import RunEKF


def test___WALTER_2015___RoboAI___ProbSet2_Q3(
        folder_path='../../MBALearnsToCode_Data/WALTER_2015___RoboAI/2.3 (EKF)/'):
    """test: WALTER (2015) "Planning, Learning & Estimation in Robotics & A.I.": Problem Set 2, Question 3"""
    control_data_file_path = folder_path + 'U.txt'
    measurement_data_file_path = folder_path + 'Z.txt'
    ground_truth_data_file_path = folder_path + 'XYT.txt'
    means_answer_file_path = folder_path + 'means.csv'
    standard_deviations_answer_file_path = folder_path + 'stdevs.csv'

    ekf = RunEKF()
    ekf.readData(control_data_file_path, measurement_data_file_path, ground_truth_data_file_path)
    ekf.run()
    means___answer = loadtxt(means_answer_file_path, delimiter=',')
    standard_deviations___answer = loadtxt(standard_deviations_answer_file_path, delimiter=',')
    assert allclose(ekf.MU, means___answer) & allclose(sqrt(ekf.VAR), standard_deviations___answer)
