from copy import deepcopy
import numpy as np


class EKF():
    # Construct an EKF instance with the following set of variables
    #    mu:                 The initial mean vector
    #    Sigma:              The initial covariance matrix
    #    R:                  The process noise covariance
    #    Q:                  The measurement noise covariance
    def __init__(self, mu, Sigma, R, Q):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q
        self.mean_state_transition_lambda = lambda m, u: m + np.array([[u[0, 0] * np.cos(m[2, 0])],
                                                                       [u[0, 0] * np.sin(m[2, 0])],
                                                                       [u[1, 0]]])
        self.mean_state_transition_jacobi_lambda = lambda m, u: np.array([[1, 0, - u[0, 0] * np.sin(m[2, 0])],
                                                                          [0, 1, u[0, 0] * np.cos(m[2, 0])],
                                                                          [0, 0, 1]])
        self.mean_observation_lambda = lambda m: np.array([[m[0, 0] ** 2 + m[1, 0] ** 2],
                                                           [m[2, 0]]])
        self.mean_observation_jacobi_lambda = lambda m, z: np.array([[2 * m[0, 0], 2 * m[1, 0], 0],
                                                                     [0, 0, 1]])

    def getMean(self):
        return self.mu

    def getCovariance(self):
        return self.Sigma

    def getVariances(self):
        return np.array([[self.Sigma[0,0],self.Sigma[1,1],self.Sigma[2,2]]])

    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u:                 The forward distance and change in heading
    def prediction(self, u):
        ekf = deepcopy(self)
        ekf.mu = ekf.mean_state_transition_lambda(ekf.mu, u)
        #print(u[0] * np.cos(ekf.mu[2]))
        #print(u[0] * np.sin(ekf.mu[2]))
        #print(u[1])
        #print(ekf.mu + np.array([[u[0] * np.cos(ekf.mu[2])],
        #                        [u[0] * np.sin(ekf.mu[2])],
        #                        [u[1]]]))
        F = ekf.mean_state_transition_jacobi_lambda(ekf.mu, u)
        ekf.Sigma = F.dot(ekf.Sigma).dot(F.T) + ekf.R
        return ekf

    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #    z:                The squared distance to the sensor and the
    #                      robot's heading
    def update(self, z):
        ekf = deepcopy(self)
        H = ekf.mean_observation_jacobi_lambda(ekf.mu, z)
        K = ekf.Sigma.dot(H.T).dot(np.linalg.pinv(H.dot(ekf.Sigma).dot(H.T) + ekf.Q))
        ekf.mu += K.dot(z - ekf.mean_observation_lambda(ekf.mu))
        ekf.Sigma = (np.eye(3) - K.dot(H)).dot(ekf.Sigma)
        return ekf