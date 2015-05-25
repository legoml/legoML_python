from __future__ import print_function
from numpy import allclose, array
from numpy.random import random
from MBALearnsToCode.Functions.FUNCTIONS___zzzMISC import approx_gradients
from MBALearnsToCode.WORK.WALTER_2015___RoboAI.RoboSoccer.RoboSoccer import distance, distance_gradients,\
    angle, angle_gradients


def UNIT_TEST___WALTER_2015___RoboAI___RoboSoccer___FunctionGradients(num_times=1000):
    num_distance_successes = 0
    num_angle_successes = 0
    for t in range(num_times):
        vector = random(4)
        distance_gradients___analytic = array(distance_gradients(*vector))
        distance_gradients___approx = approx_gradients(lambda v: distance(*v), vector)
        num_distance_successes += allclose(distance_gradients___approx, distance_gradients___analytic)
        angle_gradients___analytic = array(angle_gradients(*vector))
        angle_gradients___approx = approx_gradients(lambda v: angle(*v), vector)
        num_angle_successes += allclose(angle_gradients___approx, angle_gradients___analytic)
    print(distance_gradients___analytic)
    print(distance_gradients___approx)
    print(angle_gradients___analytic)
    print(angle_gradients___approx)
    return 100 * num_angle_successes / num_times, 100 * num_angle_successes / num_times