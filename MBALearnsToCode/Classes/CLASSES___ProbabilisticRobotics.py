from frozen_dict import FrozenDict
from sympy import log
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import gaussian_density_function as gauss_pdf
from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter


class ExtendedKalmanFilterSLAM:
    def __init__(self, ekf, mapping_jacobi_lambda):
        self.SLAM = {'': 0}
        self.EKF = ekf
        self.mapping_jacobi_lambda = mapping_jacobi_lambda
