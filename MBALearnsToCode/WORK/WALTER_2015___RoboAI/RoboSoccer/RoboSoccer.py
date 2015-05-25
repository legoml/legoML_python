from __future__ import print_function, division
from copy import copy
from numpy import arctan2, array, cos, eye, hstack, sin, vstack, zeros
from numpy.random import normal
from pandas import DataFrame
from matplotlib.pyplot import figure, imshow, plot, scatter, subplots
from matplotlib.patches import Ellipse
from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter as EKF


def euclidean_distance(x0, y0, x1, y1):
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def euclidean_distance_gradients(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    d = (dx ** 2 + dy ** 2) ** 0.5
    return - dx / d, - dy / d, dx / d, dy / d


def relative_angle(x0, y0, x1, y1):
    return arctan2(y1 - y0, x1 - x0)


def relative_angle_gradients(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    d_squared = dx ** 2 + dy ** 2
    return dy / d_squared, - dx / d_squared, - dy / d_squared, dx / d_squared


class Marking:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def same(self, marking):
        return self.id == marking.id


class Field:
    def __init__(self, length=105, width=68, goal_width=7.32):
        self.length = length
        self.width = width
        self.markings = {Marking('SouthWest', - length / 2, - width / 2),
                         Marking('NorthWest', - length / 2, width / 2),
                         Marking('SouthEast', length / 2, - width / 2),
                         Marking('NorthEast', length / 2, width / 2),
                         Marking('South', 0., - width / 2),
                         Marking('North', 0., width / 2),
                         Marking('GoalPostSW', - length / 2, - goal_width / 2),
                         Marking('GoalPostNW', - length / 2, goal_width / 2),
                         Marking('GoalPostSE', length / 2, - goal_width / 2),
                         Marking('GoalPostNE', length / 2, goal_width / 2)}
        marking_names = []
        x = []
        y = []
        for marking in self.markings:
            marking_names += [marking.id]
            x += [marking.x]
            y += [marking.y]
        self.markings_xy = DataFrame(dict(x=x, y=y), index=marking_names)


class Ball:
    def __init__(self, x, y, velocity=0, angle=0):
        self.x = x
        self.y = y
        self.x_just_now = x
        self.y_just_now = y
        self.velocity = velocity
        self.angle = angle

    def roll(self, slow_down=.9):
        self.x_just_now = copy(self.x)
        self.y_just_now = copy(self.y)
        self.x += self.velocity * cos(self.angle)
        self.y += self.velocity * sin(self.angle)
        self.velocity *= slow_down

    def kicked(self, velocity=0, angle=0):
        self.velocity = velocity
        self.angle = angle


class Player:
    def __init__(self, x, y, velocity=0, angle=0, motion_sigma=0, distance_sigma=0, team='West'):

        def transition_means(all_xy___vector, velocity_and_angle):
            new_xy_means___vector = all_xy___vector.copy()
            v, a = velocity_and_angle
            new_xy_means___vector[0] += v * cos(a)
            new_xy_means___vector[1] += v * sin(a)
            return new_xy_means___vector

        def transition_covariances(velocity_and_angle):
            return (velocity_and_angle[0] * self.motion_sigma) ** 2

        def observation_means(all_xy___vector):
            n = all_xy___vector.size
            observation_means___vector = zeros((n, 1))
            self_x, self_y = all_xy___vector[0:2, 0]
            for i in range(1, n / 2):
                k = 2 * i
                marking_x, marking_y = all_xy___vector[k:(k + 2), 0]
                observation_means___vector[k, 0] = euclidean_distance(self_x, self_y, marking_x, marking_y)
                observation_means___vector[k + 1, 0] = relative_angle(self_x, self_y, marking_x, marking_y)
            return observation_means___vector

        def observation_means_jacobi(all_xy___vector):
            n = all_xy___vector.size
            observation_means_jacobi___matrix = zeros((n, n))
            self_x, self_y = all_xy___vector[0:2, 0]
            for i in range(1, n / 2):
                k = 2 * i
                marking_x, marking_y = all_xy___vector[k:(k + 2), 0]
                observation_means_jacobi___matrix[k, (0, 1, k, k + 1)] =\
                    euclidean_distance_gradients(self_x, self_y, marking_x, marking_y)
                observation_means_jacobi___matrix[k + 1, (0, 1, k, k + 1)] =\
                    relative_angle_gradients(self_x, self_y, marking_x, marking_y)
            return observation_means_jacobi___matrix

        def observation_covariances(observations___vector):
            n = observations___vector.size
            observation_covariances___matrix = zeros((n, n))
            for i in range(1, n / 2):
                k = 2 * i
                observation_covariances___matrix[k, k] = (observations___vector[k] * self.distance_sigma) ** 2
            return observation_covariances___matrix

        self.x = x
        self.y = y
        self.velocity = velocity
        self.angle = angle
        self.motion_sigma = motion_sigma
        self.distance_sigma = distance_sigma
        self.team = team
        if team == 'West':
            self.target_goal = ('GoalPostSE', 'GoalPostNE')
        elif team == 'East':
            self.target_goal = ('GoalPostSW', 'GoalPostNW')
        self.num_mapped_markings = 0
        self.SLAM = {'self': 0}
        self.EKF = EKF(array([[x], [y]]), zeros((2, 2)),
                       transition_means, lambda all_xy___vector, velocity_and_angle: eye(all_xy___vector.size),
                       transition_covariances, observation_means, observation_means_jacobi, observation_covariances)

    def run(self):
        self.x += self.velocity * (cos(self.angle) + normal(scale=self.motion_sigma))
        self.y += self.velocity * (sin(self.angle) + normal(scale=self.motion_sigma))
        self.EKF.predict((self.velocity, self.angle))

    def augment_map(self, marking):

        mapping_jacobi = zeros((2, 2 * self.num_mapped_markings + 2))
        mapping_jacobi[0, 0] = 1.
        mapping_jacobi[1, 1] = 1.

        self.num_mapped_markings += 1
        self.SLAM[marking.id] = self.num_mapped_markings
        observed_distance = (1 + normal(scale=self.distance_sigma)) *\
            euclidean_distance(self.x, self.y, marking.x, marking.y)
        observed_angle = relative_angle(self.x, self.y, marking.x, marking.y)

        self_x_mean = self.EKF.means[0, 0]
        self_y_mean = self.EKF.means[1, 0]
        new_marking_x_mean = self_x_mean + observed_distance * cos(observed_angle)
        new_marking_y_mean = self_y_mean + observed_distance * sin(observed_angle)
        self.EKF.means = vstack((self.EKF.means, array([[new_marking_x_mean], [new_marking_y_mean]])))

        new_marking_x_own_variance = (cos(observed_angle) * observed_distance * self.distance_sigma) ** 2
        new_marking_y_own_variance = (sin(observed_angle) * observed_distance * self.distance_sigma) ** 2
        new_marking_xy_own_covariance = cos(observed_angle) * sin(observed_angle) *\
            ((observed_distance * self.distance_sigma) ** 2)
        new_marking_xy_own_covariance_matrix = array([[new_marking_x_own_variance, new_marking_xy_own_covariance],
                                                      [new_marking_xy_own_covariance, new_marking_y_own_variance]])
        new_marking_xy_covariance_matrix = mapping_jacobi.dot(self.EKF.covariances).dot(mapping_jacobi.T) +\
            new_marking_xy_own_covariance_matrix

        new_marking_xy_covariance_with_known_xy = mapping_jacobi.dot(self.EKF.covariances)

        self.EKF.covariances =\
            vstack((hstack((self.EKF.covariances, new_marking_xy_covariance_with_known_xy.T)),
                    hstack((new_marking_xy_covariance_with_known_xy, new_marking_xy_covariance_matrix))))

    def observe(self, markings):
        observations___vector = self.EKF.observation_means_lambda(self.EKF.means)
        for marking in markings:
            if marking.id in self.SLAM:
                k = 2 * self.SLAM[marking.id]
                observations___vector[k] =\
                    (1 + normal(scale=self.distance_sigma)) * euclidean_distance(self.x, self.y, marking.x, marking.y)
                observations___vector[k + 1] = relative_angle(self.x, self.y, marking.x, marking.y)
        self.EKF.update(observations___vector)
        for marking in markings:
            if marking.id not in self.SLAM:
                self.augment_map(marking)

    def orient_toward_ball(self, ball):
        self.angle = arctan2(ball.y - self.y, ball.x - self.x)

    def distance_to_ball(self, ball):
        return ((ball.x - self.x) ** 2 + (ball.y - self.y) ** 2) ** 0.5

    def have_ball(self, ball, proximity=0.1):
        return self.distance_to_ball(ball) <= proximity

    def know_goal(self):
        return bool(set(self.target_goal) & set(self.SLAM))


class Game:
    def __init__(self, field, players, ball):
        self.field = field
        self.players = players
        self.ball = ball
        self.time = 0

    def plot(self):
        return scatter(self.field.markings_xy.x, self.field.markings_xy.y, color='black', s=9)

    def out_of_play(self):
        self.ball.x > self.field.length

    def goal_scored(self):
        return True


