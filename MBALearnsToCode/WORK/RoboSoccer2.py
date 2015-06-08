from __future__ import print_function, division
from copy import deepcopy
from pprint import pprint
from time import sleep

from numpy import arctan2, array, cos, eye, hstack, pi, sin, vstack, zeros
from numpy.random import normal, uniform
from pandas import DataFrame
from matplotlib.pyplot import subplots, show
from matplotlib.patches import Ellipse

from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter as EKF
from MBALearnsToCode.WORK.ConfidenceEllipses import confidence_ellipse_parameters


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


def angular_difference(from_angle, to_angle):
    a = to_angle - from_angle
    return arctan2(sin(a), cos(a))


class Marking:
    def __init__(self, name, x, y):
        self.id = name
        self.x = x
        self.y = y


class Field:
    def __init__(self, length=105., width=68., goal_width=7.32):
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
        marking_ids = []
        x = []
        y = []
        for marking in self.markings:
            marking_ids += [marking.id]
            x += [marking.x]
            y += [marking.y]
        self.markings_xy = DataFrame(dict(x=x, y=y), index=marking_ids)


class Ball:
    def __init__(self, x=0., y=0., velocity=0., angle=0., slow_down=.9):
        self.x = x
        self.y = y
        self.x_just_now = x
        self.y_just_now = y
        self.velocity = velocity
        self.angle = angle
        self.slow_down = slow_down

    def roll(self):
        self.x_just_now = self.x
        self.y_just_now = self.y
        self.x += self.velocity * cos(self.angle)
        self.y += self.velocity * sin(self.angle)
        self.velocity *= self.slow_down

    def kicked(self, velocity=0., angle=0.):
        self.velocity = velocity
        self.angle = angle


class Player:
    def __init__(self, x=0., y=0., velocity=0., angle=0., motion_sigma=0., distance_sigma=0., team='West'):
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
                       lambda xy, v_and_a: self.transition_means(xy, v_and_a),
                       lambda xy, v_and_a: self.transition_means_jacobi(xy, v_and_a),
                       lambda v_and_a: self.transition_covariances(v_and_a),
                       lambda xy: self.observation_means(xy),
                       lambda xy: self.observation_means_jacobi(xy),
                       lambda d_and_a: self.observation_covariances(d_and_a))

    def transition_means(self, all_xy___vector, velocity_and_angle):
        conditional_xy_means___vector = deepcopy(all_xy___vector)
        v, a = velocity_and_angle
        conditional_xy_means___vector[0] += v * cos(a)
        conditional_xy_means___vector[1] += v * sin(a)
        return conditional_xy_means___vector

    def transition_means_jacobi(self, all_xy___vector, velocity_and_angle):
        return eye(2 * (self.num_mapped_markings + 1))

    def transition_covariances(self, velocity_and_angle):
        n = 2 * (self.num_mapped_markings + 1)
        conditional_xy_covariances___matrix = zeros((n, n))
        variance = (velocity_and_angle[0] * self.motion_sigma) ** 2
        conditional_xy_covariances___matrix[0, 0] = variance
        conditional_xy_covariances___matrix[1, 1] = variance
        return conditional_xy_covariances___matrix

    def observation_means(self, all_xy___vector):
        n = all_xy___vector.size
        conditional_observation_means___vector = zeros((n, 1))
        self_x, self_y = all_xy___vector[0:2, 0]
        for k in range(2, n, 2):
            marking_x, marking_y = all_xy___vector[k:(k + 2), 0]
            conditional_observation_means___vector[k, 0] = euclidean_distance(self_x, self_y, marking_x, marking_y)
            conditional_observation_means___vector[k + 1, 0] = relative_angle(self_x, self_y, marking_x, marking_y)
        return conditional_observation_means___vector

    def observation_means_jacobi(self, all_xy___vector):
        n = all_xy___vector.size
        conditional_observation_means_jacobi___matrix = zeros((n, n))
        self_x, self_y = all_xy___vector[0:2, 0]
        for k in range(2, n, 2):
            marking_x, marking_y = all_xy___vector[k:(k + 2), 0]
            conditional_observation_means_jacobi___matrix[k, (0, 1, k, k + 1)] =\
                euclidean_distance_gradients(self_x, self_y, marking_x, marking_y)
            conditional_observation_means_jacobi___matrix[k + 1, (0, 1, k, k + 1)] =\
                relative_angle_gradients(self_x, self_y, marking_x, marking_y)
        return conditional_observation_means_jacobi___matrix

    def observation_covariances(self, observations___vector):
        n = observations___vector.size
        conditional_observation_covariances___matrix = zeros((n, n))
        for k in range(2, n, 2):
            conditional_observation_covariances___matrix[k, k] = self.distance_sigma ** 2
        return conditional_observation_covariances___matrix

    def run(self):
        self.x += self.velocity * (cos(self.angle) + normal(scale=self.motion_sigma))
        self.y += self.velocity * (sin(self.angle) + normal(scale=self.motion_sigma))
        self.EKF.predict((self.velocity, self.angle))

    def augment_map(self, marking):
        print('augmenting', marking.id)
        mapping_jacobi = zeros((2, 2 * (self.num_mapped_markings + 1)))
        mapping_jacobi[0, 0] = 1.
        mapping_jacobi[1, 1] = 1.

        self.num_mapped_markings += 1
        self.SLAM[marking.id] = self.num_mapped_markings
        observed_distance = euclidean_distance(self.x, self.y, marking.x, marking.y) + normal(scale=self.distance_sigma)
        observed_angle = relative_angle(self.x, self.y, marking.x, marking.y)
        c = cos(observed_angle)
        s = sin(observed_angle)

        self_x_mean = self.EKF.means[0, 0]
        self_y_mean = self.EKF.means[1, 0]
        new_marking_x_mean = self_x_mean + c * observed_distance
        new_marking_y_mean = self_y_mean + s * observed_distance
        self.EKF.means = vstack((self.EKF.means,
                                 new_marking_x_mean,
                                 new_marking_y_mean))

        new_marking_x_own_variance = (c * self.distance_sigma) ** 2
        new_marking_y_own_variance = (s * self.distance_sigma) ** 2
        new_marking_xy_own_covariance = c * s * (self.distance_sigma ** 2)
        new_marking_xy_covariance_matrix = mapping_jacobi.dot(self.EKF.covariances).dot(mapping_jacobi.T) +\
            array([[new_marking_x_own_variance, new_marking_xy_own_covariance],
                   [new_marking_xy_own_covariance, new_marking_y_own_variance]])

        new_marking_xy_covariance_with_known_xy = mapping_jacobi.dot(self.EKF.covariances)

        self.EKF.covariances =\
            vstack((hstack((self.EKF.covariances, new_marking_xy_covariance_with_known_xy.T)),
                    hstack((new_marking_xy_covariance_with_known_xy, new_marking_xy_covariance_matrix))))

    def observe(self, markings):
        observations___vector = self.EKF.observation_means_lambda(self.EKF.means)
        for marking in markings:
            if marking.id in self.SLAM:
                k = 2 * self.SLAM[marking.id]
                observations___vector[k] = euclidean_distance(self.x, self.y, marking.x, marking.y) +\
                    normal(scale=self.distance_sigma)
                observations___vector[k + 1] = relative_angle(self.x, self.y, marking.x, marking.y)
        self.EKF.update(observations___vector)
        for marking in markings:
            if marking.id not in self.SLAM:
                self.augment_map(marking)

    def observe_marking_in_front(self, field):
        min_angle = pi
        marking_in_front = None
        for marking in field.markings:
            a = abs(angular_difference(self.angle, relative_angle(self.x, self.y, marking.x, marking.y)))
            if a < min_angle:
                min_angle = a
                marking_in_front = marking
        print('observe: ', marking_in_front.id)
        self.observe((marking_in_front,))

    def orient_randomly(self):
        self.angle = uniform(-pi, pi)

    def orient_toward_ball(self, ball):
        self.angle = arctan2(ball.y - self.y, ball.x - self.x)

    def distance_to_ball(self, ball):
        return ((ball.x - self.x) ** 2 + (ball.y - self.y) ** 2) ** 0.5

    def have_ball(self, ball, proximity=.1):
        return self.distance_to_ball(ball) <= proximity

    def know_goal(self):
        return bool(set(self.target_goal) & set(self.SLAM))


class Game:
    def __init__(self, num_players_per_team=11, velocity=1.8, motion_sigma=0.3, distance_sigma=1.):
        self.field = Field()
        length = self.field.length
        width = self.field.width
        self.num_players_per_team = num_players_per_team
        self.players = []
        for i in range(num_players_per_team):
            self.players += [Player(uniform(- length / 2, 0), uniform(- width / 2, width / 2),
                                    velocity, uniform(-pi, pi), motion_sigma, distance_sigma, 'West'),
                             Player(uniform(0, length / 2), uniform(- width / 2, width / 2),
                                    velocity, uniform(-pi, pi), motion_sigma, distance_sigma, 'East')]
        self.ball = Ball()
        self.time = 0

        self.figure, self.game_plot = subplots()
        self.boundary_plot = None
        self.markings_plot = None
        self.ball_plot = None
        self.west_team_plot = None
        self.east_team_plot = None
        self.markings_being_observed = None
        self.SLAM_confidence_plots = {}
        self.SLAM_arrow_plots = {}

    def plot_init(self):
        length = self.field.length
        width = self.field.width
        self.boundary_plot, = self.game_plot.plot(
            (- length / 2, length / 2, length / 2, - length / 2, - length / 2),
            (- width / 2, - width / 2, width / 2, width / 2, - width / 2), color='gray', linewidth=1)
        self.markings_plot = self.game_plot.scatter(
            self.field.markings_xy.x, self.field.markings_xy.y, color='black', s=24)
        west_x = []
        west_y = []
        east_x = []
        east_y = []
        for i in range(self.num_players_per_team):
            k = 2 * i
            west_x += [self.players[k].x]
            west_y += [self.players[k].y]
            east_x += [self.players[k + 1].x]
            east_y += [self.players[k + 1].y]
        self.ball_plot = self.game_plot\
            .scatter(self.ball.x, self.ball.y, s=36, color='magenta', marker='o') #, animated=True
        self.west_team_plot = self.game_plot\
            .scatter(west_x, west_y, s=48, c='blue', marker='o', label='West')  #, animated=True
        self.east_team_plot = self.game_plot\
            .scatter(east_x, east_y, s=48, c='red', marker='o', label='East')   #, animated=True
        show()

    def update_ball_and_player_plots(self):
        self.players_run_randomly()
        west_xy = zeros((2, self.num_players_per_team))
        east_xy = zeros((2, self.num_players_per_team))
        for i in range(self.num_players_per_team):
            k = 2 * i
            west_xy[0, i] = self.players[k].x
            west_xy[1, i] = self.players[k].y
            east_xy[0, i] = self.players[k + 1].x
            east_xy[1, i] = self.players[k + 1].y
        self.ball_plot.set_offsets(array([[self.ball.x], [self.ball.y]]))
        self.west_team_plot.set_offsets(west_xy)
        self.east_team_plot.set_offsets(east_xy)
        for obj, i in self.players[0].SLAM.items():
            k = 2 * i
            if i:
                marking_x = self.field.markings_xy.ix[obj].x
                marking_y = self.field.markings_xy.ix[obj].y
            xy = self.players[0].EKF.means[k:(k + 2), 0]
            width, height, angle =\
                confidence_ellipse_parameters(self.players[0].EKF.covariances[k:(k + 2), k:(k + 2)])
            if obj in self.SLAM_confidence_plots:
                if i:
                    self.SLAM_arrow_plots[obj].set_xdata([self.players[0].x, marking_x])
                    self.SLAM_arrow_plots[obj].set_ydata([self.players[0].y, marking_y])
                self.SLAM_confidence_plots[obj].center = xy
                self.SLAM_confidence_plots[obj].width = width
                self.SLAM_confidence_plots[obj].height = height
                self.SLAM_confidence_plots[obj].angle = angle
            else:
                if i:
                    self.SLAM_arrow_plots[obj], = self.game_plot.plot(
                        [self.players[0].x, marking_x], [self.players[0].y, marking_y],
                        color='lightgray', animated=True)
                self.SLAM_confidence_plots[obj] = self.game_plot.add_artist(
                    Ellipse(xy=xy, width=width, height=height, angle=angle, facecolor='green', edgecolor='green',
                            alpha=0.6, animated=True))
        #self.game_plot.relim()
        #self.game_plot.autoscale_view(True, True, True)
        self.figure.canvas.draw()
        #self.figure.canvas.flush_events()
        sleep(1e-6)

    def out_of_play(self):
        return (self.ball.x < -self.field.length / 2) | (self.ball.x > self.field.length / 2) |\
            (self.ball.y < -self.field.width / 2) | (self.ball.y > self.field.width / 2)

    def goal_scored(self):
        return True

    def players_run_randomly(self):
        self.time += 1
        for i in range(0, 2 * self.num_players_per_team):
            self.players[i].orient_randomly()
            #self.players[i].orient_toward_ball(self.ball)
            self.players[i].play()
            self.players[i].observe_marking_in_front(self.field)
            #self.players[i].observe(self.field.markings)
        pprint(self.players[0].SLAM)
        pprint(self.players[0].EKF.covariances)

    def play(self):
        self.plot_init()
        while not self.out_of_play():
            self.time += 1
            for i in range(0, 2 * self.num_players_per_team):
                self.players[i].orient_randomly()
                self.players[i].play()
                self.players[i].observe_marking_in_front(self.field)
            #self.update_ball_and_player_plots()
            pprint(self.players[0].SLAM)
            pprint(self.players[0].EKF.covariances)

