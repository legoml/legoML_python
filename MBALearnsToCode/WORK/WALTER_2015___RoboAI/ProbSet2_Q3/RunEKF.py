from __future__ import print_function
import matplotlib.pyplot as plt
from numpy import allclose, array, atleast_2d, concatenate, cos, degrees, diag, loadtxt, pi, radians, sin, size, sqrt,\
    unwrap, zeros
import sys
from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter as EKF


class RunEKF:
    def __init__(self):
        self.R = array([[2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, radians(2)]])*1e-6   #1E-4
        self.Q = array([[1.0, 0.0],[0.0, radians(1)]])*1E-8   #1E-6
        self.U = [] # Array that stores the control data where rows increase with time
        self.Z = [] # Array that stores the measurement data where rows increase with time
        self.XYT = [] # Array that stores the ground truth pose where rows increase with time
        self.MU = [] # Array in which to append mu_t as a row vector after each iteration
        self.VAR = [] # Array in which to append var(x), var(y), var(theta)
                      # from the EKF covariance as a row vector after each iteration

    # Read in the control and measurement data from their respective text files
    # and populates self.U and self.Z
    def readData(self, filenameU, filenameZ, filenameXYT):
        print("Reading control data from %s and measurement data from %s" % (filenameU, filenameZ))
        self.U = loadtxt(filenameU, comments='#', delimiter=',').astype(float)
        self.Z = loadtxt(filenameZ, comments='#', delimiter=',').astype(float)
        self.XYT = loadtxt(filenameXYT, comments='#', delimiter=',').astype(float)

    # Iterate from t=1 to t=T performing the two filtering steps
    def run(self):
        mu0 = array([[-4.0], [-4.0], [pi/2]])
        Sigma0 = zeros((3, 3))
        self.VAR = array([[Sigma0[0, 0], Sigma0[1, 1], Sigma0[2, 2]]])
        self.MU = mu0.T

        transition_means_lambda = lambda m, u: m + array([[u[0, 0] * cos(m[2, 0])],
                                                          [u[0, 0] * sin(m[2, 0])],
                                                          [u[1, 0]]])
        transition_means_jacobi_lambda = lambda m, u: array([[1, 0, - u[0, 0] * sin(m[2, 0])],
                                                             [0, 1, u[0, 0] * cos(m[2, 0])],
                                                             [0, 0, 1]])
        transition_covariances_lambda = lambda u: self.R
        observation_means_lambda = lambda m: array([[m[0, 0] ** 2 + m[1, 0] ** 2],
                                                    [m[2, 0]]])
        observation_means_jacobi_lambda = lambda m: array([[2 * m[0, 0], 2 * m[1, 0], 0],
                                                           [0, 0, 1]])
        observation_covariances_lambda = lambda z: self.Q
        self.ekf = EKF(mu0, Sigma0,
                       transition_means_lambda, transition_means_jacobi_lambda, transition_covariances_lambda,
                       observation_means_lambda, observation_means_jacobi_lambda, observation_covariances_lambda)

        # For each t in [1,T]
        #    1) Call self.ekf.prediction(u_t)
        #    2) Call self.ekf.update(z_t)
        #    3) Add self.ekf.getMean() to self.MU
        T = self.U.shape[0]
        for t in range(T):
            self.ekf.predict(atleast_2d(self.U[t, :]).T)
            self.ekf.update(atleast_2d(self.Z[t, :]).T)
            self.MU = concatenate((self.MU, self.ekf.means.T))
            self.VAR = concatenate((self.VAR, atleast_2d(diag(self.ekf.covariances))))

        print('Final Mean State Estimate:')
        print(self.ekf.means)
        print('Final Covariance State Estimate:')
        print(self.ekf.covariances)

    # Plot the resulting estimate for the robot's trajectory
    def plot(self):

        # Plot the estimated and ground truth trajectories
        ground_truth = plt.plot(self.XYT[:, 0], self.XYT[:, 1], 'g.-', label='Ground Truth')
        mean_trajectory = plt.plot(self.MU[:, 0], self.MU[:, 1], 'r.-', label='Estimate')
        plt.legend()

        # Try changing this to different standard deviations
        sigma = 1 # 2 or 3

        # Plot the errors with error bars
        Error = self.XYT-self.MU
        T = range(size(self.XYT,0))
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(T,Error[:,0],'r-')
        axarr[0].plot(T,sigma*sqrt(self.VAR[:,0]),'b--')
        axarr[0].plot(T,-sigma*sqrt(self.VAR[:,0]),'b--')
        axarr[0].set_title('X error')
        axarr[0].set_ylabel('Error (m)')
        
        axarr[1].plot(T,Error[:,1],'r-')
        axarr[1].plot(T,sigma*sqrt(self.VAR[:,1]),'b--')
        axarr[1].plot(T,-sigma*sqrt(self.VAR[:,1]),'b--')
        axarr[1].set_title('Y error')
        axarr[1].set_ylabel('Error (m)')
        
        axarr[2].plot(T,degrees(unwrap(Error[:,2])),'r-')
        axarr[2].plot(T,sigma*degrees(unwrap(sqrt(self.VAR[:,2]))),'b--')
        axarr[2].plot(T,-sigma*degrees(unwrap(sqrt(self.VAR[:,2]))),'b--')
        axarr[2].set_title('Theta error (degrees)')
        axarr[2].set_ylabel('Error (degrees)')
        axarr[2].set_xlabel('Time')
        
        plt.show()


def UNIT_TEST___WALTER_2015___RoboAI___ProbSet2_Q3(control_data_file_path, measurement_data_file_path,
                                                   ground_truth_data_file_path, means_answer_file_path,
                                                   standard_deviations_answer_file_path):
    ekf = RunEKF()
    ekf.readData(control_data_file_path, measurement_data_file_path, ground_truth_data_file_path)
    ekf.run()
    ekf.plot()
    means___answer = loadtxt(means_answer_file_path, delimiter=',')
    standard_deviations___answer = loadtxt(standard_deviations_answer_file_path, delimiter=',')
    return allclose(ekf.MU, means___answer) & allclose(sqrt(ekf.VAR), standard_deviations___answer)


if __name__ == '__main__':

    # This function should be called with three arguments:
    #    sys.argv[1]: Comma-delimited file containing control data (U.txt)
    #    sys.argv[2]: Comma-delimited file containing measurement data (Z.txt)
    #    sys.argv[3]: Comma-delimited file containing ground-truth poses (XYT.txt)
    if len(sys.argv) != 4:
        print("usage: RunEKF.py ControlData.txt MeasurementData.txt GroundTruthData.txt")
        sys.exit(2)
        
    ekf = RunEKF()
    ekf.readData(sys.argv[1], sys.argv[2], sys.argv[3])
    ekf.run()
    ekf.plot()