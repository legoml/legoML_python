from HelpyFuncs.Geometry2D import euclidean_distance, euclidean_distance_gradients, ray_angle, ray_angle_gradients
from HelpyFuncs.zzz import approx_gradients
from numpy import allclose, array
from numpy.random import random


def test___Geometry2D_Distances_and_Angles(num_times=1000):
    """test: Gradients of Euclidean Distance and Ray Angle Functions in 2D Geometry"""
    num_distance_successes = 0
    num_angle_successes = 0
    for t in range(num_times):
        vector = 1000 * random(4)
        distance_gradients___analytic = array(euclidean_distance_gradients(*vector))
        distance_gradients___approx = approx_gradients(lambda v: euclidean_distance(*v), vector)
        num_distance_successes += allclose(distance_gradients___approx, distance_gradients___analytic)
        angle_gradients___analytic = array(ray_angle_gradients(*vector))
        angle_gradients___approx = approx_gradients(lambda v: ray_angle(*v), vector)
        num_angle_successes += allclose(angle_gradients___approx, angle_gradients___analytic)
    assert (num_angle_successes == num_times) & (num_angle_successes / num_times)
