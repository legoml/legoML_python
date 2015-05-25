from __future__ import print_function
from numpy import allclose, array
from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter as EKF


def UNIT_TEST___WALTER_2015___RoboAI___MidTerm_Q3():

    means = array([[0.]])
    covariances = array([[0.]])
    transition_means_lambda = lambda mu, control: mu + control
    transition_means_jacobi_lambda = lambda mu, control: array([[1.]])
    transition_covariances_lambda = lambda control: array([[0.1 ** 2]])
    observation_means_lambda = lambda state: state
    observation_means_jacobi_lambda = lambda state: array([[1.]])
    observation_covariances_lambda = lambda observation: array([[0.5 ** 2]])
    ekf = EKF(means, covariances,
              transition_means_lambda, transition_means_jacobi_lambda, transition_covariances_lambda,
              observation_means_lambda, observation_means_jacobi_lambda, observation_covariances_lambda)

    ekf.predict(1.)
    print('Predict 1:', ekf.means, ekf.covariances, ekf.standard_deviations())
    mu_predicted_1 = array([[1.]])
    sigma_squared_predicted_1 = array([[.01]])
    predicted_1___check = allclose(ekf.means, mu_predicted_1) &\
        allclose(ekf.covariances, sigma_squared_predicted_1)

    ekf.update(1.2)
    print('Update 1:', ekf.means, ekf.covariances, ekf.standard_deviations())
    mu_updated_1 = array([[1.0077]])
    sigma_squared_updated_1 = array([[.0096]])
    sigma_updated_1 = array([[.0981]])
    updated_1___check = allclose(ekf.means, mu_updated_1) &\
        allclose(ekf.covariances, sigma_squared_updated_1, atol=1e-4) &\
        allclose(ekf.standard_deviations(), sigma_updated_1, atol=1e-4)

    ekf.predict(1.)
    print('Predict 2:', ekf.means, ekf.covariances, ekf.standard_deviations())
    mu_predicted_2 = array([[2.0077]])
    sigma_squared_predicted_2 = array([[.0196]]),
    predicted_2___check = allclose(ekf.means, mu_predicted_2) &\
        allclose(ekf.covariances, sigma_squared_predicted_2, atol=1e-4)

    ekf.update(1.5)
    print('Update 2:', ekf.means, ekf.covariances, ekf.standard_deviations())
    mu_updated_2 = array([[1.9708]])
    sigma_squared_updated_2 = array([[.0182]])
    sigma_updated_2 = array([[.1349]])
    updated_2___check = allclose(ekf.means, mu_updated_2, atol=1e-4) &\
        allclose(ekf.covariances, sigma_squared_updated_2, atol=1e-4) &\
        allclose(ekf.standard_deviations(), sigma_updated_2, atol=1e-4)

    ekf.predict(1.)
    print('Predict 3:', ekf.means, ekf.covariances, ekf.standard_deviations())

    ekf.update(2.5)
    print('Update 3:', ekf.means, ekf.covariances, ekf.standard_deviations())
    mu_updated_3 = array([[2.9231]])
    sigma_updated_3 = array([[.1592]])
    updated_3___check = allclose(ekf.means, mu_updated_3, atol=1e-4) &\
        allclose(ekf.standard_deviations(), sigma_updated_3, atol=1e-4)

    ekf.predict(1.)
    print('Predict 4:', ekf.means, ekf.covariances, ekf.standard_deviations())

    ekf.update(4.5)
    print('Update 4:', ekf.means, ekf.covariances, ekf.standard_deviations())
    mu_updated_4 = array([[3.9945]])
    sigma_updated_4 = array([[.1759]])
    updated_4___check = allclose(ekf.means, mu_updated_4, atol=1e-4) &\
        allclose(ekf.standard_deviations(), sigma_updated_4, atol=1e-4)

    ekf.predict(1.)
    print('Predict 5:', ekf.means, ekf.covariances, ekf.standard_deviations())

    ekf.update(6.)
    print('Update 5:', ekf.means, ekf.covariances, ekf.standard_deviations())
    mu_updated_5 = array([[5.136]])
    sigma_updated_5 = array([[.1876]])
    updated_5___check = allclose(ekf.means, mu_updated_5) &\
        allclose(ekf.standard_deviations(), sigma_updated_5, atol=1e-5)

    return predicted_1___check & updated_1___check & predicted_2___check & updated_2___check &\
        updated_3___check & updated_4___check & updated_5___check