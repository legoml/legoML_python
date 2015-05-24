from numpy import diag, sqrt
from numpy.linalg import pinv

class ExtendedKalmanFilter:
    def __init__(self, state_means, state_covariances, transition_covariances, observation_covariances,
                 state_means_transition_lambda, state_means_transition_jacobi_lambda,
                 observation_means_lambda, observation_means_jacobi_lambda):
        self.means = state_means
        self.covariances = state_covariances
        self.transition_covariances = transition_covariances
        self.observation_covariances = observation_covariances
        self.state_means_transition_lambda = state_means_transition_lambda
        self.state_means_transition_jacobi_lambda = state_means_transition_jacobi_lambda
        self.observation_means_lambda = observation_means_lambda
        self.observation_means_jacobi_lambda = observation_means_jacobi_lambda

    def standard_deviations(self):
        return sqrt(diag(self.covariances))

    def predict(self, control_data):
        self.means = self.state_means_transition_lambda(self.means, control_data)
        F = self.state_means_transition_jacobi_lambda(self.means, control_data)
        self.covariances = F.dot(self.covariances).dot(F.T) + self.transition_covariances

    def update(self, observation_data):
        H = self.observation_means_jacobi_lambda(self.means)
        K = self.covariances.dot(H.T).dot(pinv(H.dot(self.covariances).dot(H.T) + self.observation_covariances))
        self.means += K.dot(observation_data - self.observation_means_lambda(self.means))
        self.covariances -= K.dot(H).dot(self.covariances)