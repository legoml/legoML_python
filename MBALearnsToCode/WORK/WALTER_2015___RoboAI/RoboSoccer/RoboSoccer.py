from copy import copy
from numpy import arctan2, array, cos, eye, sin
from numpy.random import normal
from pandas import DataFrame
from matplotlib.pyplot import figure, imshow, plot, scatter, subplots
from matplotlib.patches import Ellipse
from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter as EKF


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

    def is_kicked(self, velocity, angle):
        self.velocity = velocity
        self.angle = angle


class Player:
    def __init__(self, x, y, velocity=0, angle=0, motion_sigma=0, distance_sigma=0, angle_sigma=0, team='West'):

        def transition_means(state_means, velocity_and_angle):
            new_state_means = state_means.copy()
            v, a = velocity_and_angle
            new_state_means[0] += v * cos(a)
            new_state_means[1] += v * sin(a)
            return new_state_means

        def transition_covariances(velocity_and_angle):
            v = velocity_and_angle[0]
            return self.motion_sigma * v

        def observation_means():
            return True

        self.x = x
        self.y = y
        self.velocity = velocity
        self.angle = angle
        self.motion_sigma = motion_sigma
        self.distance_sigma = distance_sigma
        self.angle_sigma = angle_sigma
        self.team = team
        if team == 'West':
            self.target_goal = ('GoalPostSE', 'GoalPostNE')
        elif team == 'East':
            self.target_goal = ('GoalPostSW', 'GoalPostNW')

        init_means = array([[x], [y]])
        init_covariances = array([[0., 0.], [0., 0.]])
        transition_means_lambda = lambda means, velocity_and_angle: transition_means(means, velocity_and_angle)
        transition_means_jacobi_lambda = lambda means, velocity_and_angle: eye(means.size)
        transition_covariances_lambda = lambda velocity_and_angle: transition_covariances(velocity_and_angle)
        observation_means_lambda = lambda: None
        observation_means_jacobi_lambda = lambda: None
        observation_covariances_lambda = lambda: None
        self.EKF = EKF(init_means, init_covariances,
                       transition_means_lambda, transition_means_jacobi_lambda, transition_covariances_lambda,
                       observation_means_lambda, observation_means_jacobi_lambda, observation_covariances_lambda)

    def run(self):
        self.x += self.velocity * (cos(self.angle) + normal(scale=self.motion_sigma))
        self.y += self.velocity * (sin(self.angle) + normal(scale=self.motion_sigma))
        self.EKF.predict((self.velocity, self.angle))

    def observe(self, markings):
        self.EKF.update(None)

    def orient_toward_ball(self, ball):
        self.angle = arctan2(ball.y - self.y, ball.x - self.x)

    def distance_to_ball(self, ball):
        return ((ball.x - self.x) ** 2 + (ball.y - self.y) ** 2) ** 0.5

    def have_ball(self, ball, proximity=0.1):
        return self.distance_to_ball(ball) <= proximity

    def know_goal(self):
        return bool(set(self.target_goal) & set(self.SLAM_belief))


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


def distance(x0, y0, x1, y1):
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def distance_gradients(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    d = (dx ** 2 + dy ** 2) ** 0.5
    return - dx / d, - dy / d, dx / d, dy / d


def angle(x0, y0, x1, y1):
    return arctan2(y1 - y0, x1 - x0)


def angle_gradients(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    d_squared = dx ** 2 + dy ** 2
    return dy / d_squared, - dx / d_squared, - dy / d_squared, dx / d_squared