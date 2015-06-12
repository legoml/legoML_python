from __future__ import print_function, division
#from matplotlib import use
#use('TkAgg')

from copy import deepcopy
from numpy import abs, arctan2, array, atleast_2d, ceil, cos, eye, floor, hstack, inf, nan, pi, sin, vstack, zeros
from numpy.random import normal, uniform
from pandas import DataFrame
from pprint import pprint
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.pyplot import figure, imshow, subplot
from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter as EKF
from MBALearnsToCode.Functions.FUNCTIONS___Geometry2D import euclidean_distance, euclidean_distance_gradients,\
    ray_angle, ray_angle_gradients, angular_difference
from MBALearnsToCode.Functions.FUNCTIONS___Visualizations import gaussian_confidence_ellipse_parameters
from MBALearnsToCode.Functions.FUNCTIONS___zzzUtility import within_range


class Marking:
    """MARKING class

    Data structure for fixed markings on soccer field
    """
    def __init__(self, name, x, y):
        self.id = name
        self.x = x
        self.y = y


class Field:
    def __init__(self, length=105., width=68., goal_width=7.32):
        self.length = length
        self.width = width
        self.goal_width = goal_width
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
    def __init__(self, x=0., y=0., velocity=0., angle=0., slow_down=.8):
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
    def __init__(self, x=0., y=0., velocity=0., angle=0., acceleration_sigma=0., motion_sigma=0., distance_sigma=0.,
                 angle_sigma=0., inconsistency_threshold=9., team='West'):
        self.x = x
        self.y = y
        self.velocity = velocity
        self.angle = angle
        self.acceleration_sigma = acceleration_sigma
        self.motion_sigma = motion_sigma
        self.distance_sigma = distance_sigma
        self.angle_sigma = angle_sigma
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
                       lambda xy, marking_indices=tuple():
                           self.observation_means(xy, marking_indices=marking_indices),
                       lambda xy, marking_indices=tuple():
                           self.observation_means_jacobi(xy, marking_indices=marking_indices),
                       lambda d_and_a, marking_indices=tuple():
                           self.observation_covariances(d_and_a, marking_indices=marking_indices))
        self.inconsistency_threshold = inconsistency_threshold
        self.lost = False

    def transition_means(self, current_xy___vector, velocity_and_angle):
        conditional_next_xy_means___vector = deepcopy(current_xy___vector)
        v, a = velocity_and_angle
        conditional_next_xy_means___vector[0, 0] += v * cos(a)
        conditional_next_xy_means___vector[1, 0] += v * sin(a)
        return conditional_next_xy_means___vector

    def transition_means_jacobi(self, all_xy___vector, velocity_and_angle):
        return eye(2 * (self.num_mapped_markings + 1))

    def transition_covariances(self, velocity_and_angle):
        n = 2 * (self.num_mapped_markings + 1)
        conditional_next_xy_covariances___matrix = zeros((n, n))
        variance = (velocity_and_angle[0] * self.motion_sigma) ** 2
        conditional_next_xy_covariances___matrix[0, 0] = variance
        conditional_next_xy_covariances___matrix[1, 1] = variance
        return conditional_next_xy_covariances___matrix

    def observation_means(self, xy___vector, marking_indices=tuple()):
        n_markings = len(marking_indices)
        conditional_observation_means___vector = zeros((2 * n_markings, 1))
        self_x, self_y = xy___vector[0:2, 0]
        for i in range(n_markings):
            k = 2 * i
            k_marking = 2 * marking_indices[i]
            marking_x, marking_y = xy___vector[k_marking:(k_marking + 2), 0]
            conditional_observation_means___vector[k, 0] = euclidean_distance(self_x, self_y, marking_x, marking_y)
            conditional_observation_means___vector[k + 1, 0] = ray_angle(self_x, self_y, marking_x, marking_y)
        return conditional_observation_means___vector

    def observation_means_jacobi(self, xy___vector, marking_indices=tuple()):
        n_markings = len(marking_indices)
        conditional_observation_means_jacobi___matrix = zeros((2 * n_markings, xy___vector.size))
        self_x, self_y = xy___vector[0:2, 0]
        for i in range(n_markings):
            k = 2 * i
            k_marking = 2 * marking_indices[i]
            marking_x, marking_y = xy___vector[k_marking:(k_marking + 2), 0]
            conditional_observation_means_jacobi___matrix[k, (0, 1, k_marking, k_marking + 1)] =\
                euclidean_distance_gradients(self_x, self_y, marking_x, marking_y)
            conditional_observation_means_jacobi___matrix[k + 1, (0, 1, k_marking, k_marking + 1)] =\
                ray_angle_gradients(self_x, self_y, marking_x, marking_y)
        return conditional_observation_means_jacobi___matrix

    def observation_covariances(self, observations___vector, marking_indices=tuple()):
        n_markings = len(marking_indices)
        conditional_observation_covariances___matrix = zeros((2 * n_markings, 2 * n_markings))
        for i in range(n_markings):
            k = 2 * i
            conditional_observation_covariances___matrix[k, k] = self.distance_sigma ** 2
            conditional_observation_covariances___matrix[k + 1, k + 1] = self.angle_sigma ** 2
        return conditional_observation_covariances___matrix

    def accelerate(self, acceleration):
        v = self.velocity + acceleration
        if v > 0:
            self.velocity = v

    def run(self):
        self.x += self.velocity * (cos(self.angle) + normal(scale=self.motion_sigma / 2))  # to avoid over-confidence
        self.y += self.velocity * (sin(self.angle) + normal(scale=self.motion_sigma / 2))  # to avoid over-confidence
        self.EKF.predict((self.velocity, self.angle))

    def augment_map(self, marking):

        mapping_jacobi = zeros((2, 2 * (self.num_mapped_markings + 1)))
        mapping_jacobi[0, 0] = 1.
        mapping_jacobi[1, 1] = 1.

        self.num_mapped_markings += 1
        self.SLAM[marking.id] = self.num_mapped_markings

        observed_distance = euclidean_distance(self.x, self.y, marking.x, marking.y) +\
            normal(scale=self.distance_sigma / 2)  # to avoid over-confidence
        distance_minus_1sd = observed_distance - self.distance_sigma
        distance_plus_1sd = observed_distance + self.distance_sigma

        observed_angle = ray_angle(self.x, self.y, marking.x, marking.y) +\
            normal(scale=self.angle_sigma / 2)  # to avoid over-confidence
        c = cos(observed_angle)
        s = sin(observed_angle)
        angle_minus_1sd = observed_angle - self.angle_sigma
        c_minus = cos(angle_minus_1sd)
        s_minus = sin(angle_minus_1sd)
        angle_plus_1sd = observed_angle + self.angle_sigma
        c_plus = cos(angle_plus_1sd)
        s_plus = sin(angle_plus_1sd)

        self_x_mean = self.EKF.means[0, 0]
        self_y_mean = self.EKF.means[1, 0]
        new_marking_x_mean = self_x_mean + c * observed_distance
        new_marking_y_mean = self_y_mean + s * observed_distance
        self.EKF.means = vstack((self.EKF.means,
                                 new_marking_x_mean,
                                 new_marking_y_mean))

        x_minus_distance_1sd_minus_angle_1sd = self_x_mean + c_minus * distance_minus_1sd
        y_minus_distance_1sd_minus_angle_1sd = self_y_mean + s_minus * distance_minus_1sd
        x_minus_distance_1sd_same_angle = self_x_mean + c * distance_minus_1sd
        y_minus_distance_1sd_same_angle = self_y_mean + s * distance_minus_1sd
        x_minus_distance_1sd_plus_angle_1sd = self_x_mean + c_plus * distance_minus_1sd
        y_minus_distance_1sd_plus_angle_1sd = self_y_mean + s_plus * distance_minus_1sd

        x_same_distance_minus_angle_1sd = self_x_mean + c_minus * observed_distance
        y_same_distance_minus_angle_1sd = self_y_mean + s_minus * observed_distance
        x_same_distance_plus_angle_1sd = self_x_mean + c_plus * observed_distance
        y_same_distance_plus_angle_1sd = self_y_mean + s_plus * observed_distance

        x_plus_distance_1sd_minus_angle_1sd = self_x_mean + c_minus * distance_plus_1sd
        y_plus_distance_1sd_minus_angle_1sd = self_y_mean + s_minus * distance_plus_1sd
        x_plus_distance_1sd_same_angle = self_x_mean + c * distance_plus_1sd
        y_plus_distance_1sd_same_angle = self_y_mean + s * distance_plus_1sd
        x_plus_distance_1sd_plus_angle_1sd = self_x_mean + c_plus * distance_plus_1sd
        y_plus_distance_1sd_plus_angle_1sd = self_y_mean + s_plus * distance_plus_1sd

        x_1sd = array((x_minus_distance_1sd_minus_angle_1sd, x_minus_distance_1sd_same_angle,
                       x_minus_distance_1sd_plus_angle_1sd, x_same_distance_minus_angle_1sd,
                       x_same_distance_plus_angle_1sd, x_plus_distance_1sd_minus_angle_1sd,
                       x_plus_distance_1sd_same_angle, x_plus_distance_1sd_plus_angle_1sd))
        y_1sd = array((y_minus_distance_1sd_minus_angle_1sd, y_minus_distance_1sd_same_angle,
                       y_minus_distance_1sd_plus_angle_1sd, y_same_distance_minus_angle_1sd,
                       y_same_distance_plus_angle_1sd, y_plus_distance_1sd_minus_angle_1sd,
                       y_plus_distance_1sd_same_angle, y_plus_distance_1sd_plus_angle_1sd))

        new_marking_x_own_standard_deviation = max(abs(x_1sd - new_marking_x_mean))
        new_marking_x_own_variance = new_marking_x_own_standard_deviation ** 2
        new_marking_y_own_standard_deviation = max(abs(y_1sd - new_marking_y_mean))
        new_marking_y_own_variance = new_marking_y_own_standard_deviation ** 2
        new_marking_xy_own_covariance = c * s * (self.distance_sigma ** 2)
        new_marking_xy_covariance_matrix = mapping_jacobi.dot(self.EKF.covariances).dot(mapping_jacobi.T) +\
            array([[new_marking_x_own_variance, new_marking_xy_own_covariance],
                   [new_marking_xy_own_covariance, new_marking_y_own_variance]])

        new_marking_xy_covariance_with_known_xy = mapping_jacobi.dot(self.EKF.covariances)

        self.EKF.covariances =\
            vstack((hstack((self.EKF.covariances, new_marking_xy_covariance_with_known_xy.T)),
                    hstack((new_marking_xy_covariance_with_known_xy, new_marking_xy_covariance_matrix))))

    def observe(self, markings, min_distance_over_distance_sigma_ratio=6.):
        observations___vector = array([[]]).T
        marking_indices = []
        for marking in markings:
            if marking.id in self.SLAM:
                d = euclidean_distance(self.x, self.y, marking.x, marking.y)
                if d >= min_distance_over_distance_sigma_ratio * self.distance_sigma:
                    observed_distance = d + normal(scale=self.distance_sigma / 2)  # to avoid over-confidence
                    observed_angle = ray_angle(self.x, self.y, marking.x, marking.y) +\
                        normal(scale=self.angle_sigma / 2)   # to avoid over-confidence
                    observations___vector = vstack((observations___vector,
                                                    observed_distance,
                                                    observed_angle))
                    marking_indices += [self.SLAM[marking.id]]
        if marking_indices:
            current_means = deepcopy(self.EKF.means)
            self.EKF.update(observations___vector, marking_indices=marking_indices)
            if euclidean_distance(current_means[0, 0], current_means[1, 0],
                                  self.EKF.means[0, 0], self.EKF.means[1, 0]) > self.inconsistency_threshold:
                self.lost = True

        for marking in markings:
            if marking.id not in self.SLAM:
                self.augment_map(marking)

    def observe_marking_in_front(self, field, min_distance_over_distance_sigma_ratio=6.):
        min_angle = pi
        marking_in_front = None
        for marking in field.markings:
            a = abs(angular_difference(self.angle, ray_angle(self.x, self.y, marking.x, marking.y)))
            d = euclidean_distance(self.x, self.y, marking.x, marking.y)
            if (a < min_angle) and (d >= min_distance_over_distance_sigma_ratio * self.distance_sigma):
                min_angle = a
                marking_in_front = marking
        self.observe((marking_in_front,))

    def orient_randomly(self):
        self.angle = uniform(-pi, pi)

    def orient_toward_ball(self, ball):
        self.angle = arctan2(ball.y - self.y, ball.x - self.x)

    def distance_to_ball(self, ball):
        return ((ball.x - self.x) ** 2 + (ball.y - self.y) ** 2) ** 0.5

    def know_goal(self):
        return bool(set(self.target_goal) & set(self.SLAM))


class Game:
    def __init__(self, num_players=6, velocity=6., acceleration_sigma=3.,
                 motion_sigma=0.3, distance_sigma=1., angle_sigma=3 * pi / 180, inconsistency_threshold=3.,
                 ball_kick_velocity=3., ball_slow_down=0.8, team_names=('Chelsea', 'Arsenal')):
        self.field = Field()
        length = self.field.length
        width = self.field.width
        self.num_players = num_players
        self.players = []
        for i in range(num_players):
            if i % 2:
                self.players += [Player(x=uniform(0, length / 2), y=uniform(- width / 2, width / 2),
                                        velocity=velocity, angle=uniform(-pi, pi),
                                        acceleration_sigma=acceleration_sigma, motion_sigma=motion_sigma,
                                        distance_sigma=distance_sigma, angle_sigma=angle_sigma,
                                        inconsistency_threshold=inconsistency_threshold, team='East')]
            else:
                self.players += [Player(x=uniform(- length / 2, 0), y=uniform(- width / 2, width / 2),
                                        velocity=velocity, angle=uniform(-pi, pi),
                                        acceleration_sigma=acceleration_sigma, motion_sigma=motion_sigma,
                                        distance_sigma=distance_sigma, angle_sigma=angle_sigma,
                                        inconsistency_threshold=inconsistency_threshold, team='West')]

        self.ball_kick_velocity = ball_kick_velocity
        self.ball_slow_down = ball_slow_down
        self.ball = Ball(slow_down=ball_slow_down)

        self.time = 0
        self.west_team, self.east_team = team_names
        self.score = {'West': 0, 'East': 0}

        self.figure = figure()
        gs = GridSpec(1, 2, width_ratios=[5, 1])
        self.game_plot = subplot(gs[0])
        self.SLAM_plot = subplot(gs[1])
        self.boundary_plot = None
        self.goal_box_w_plot = None
        self.goal_box_e_plot = None
        self.markings_plot = None
        self.ball_plot = None
        self.west_team_plot = None
        self.east_team_plot = None
        self.west_score_text_plot = None
        self.east_score_text_plot = None
        self.markings_being_observed = None
        self.SLAM_confidence_plots = {}
        self.SLAM_arrow_plots = {}
        self.cov_plot = None
        self.beliefs = self.field.markings_xy
        self.beliefs['bias_x'] = float(nan)
        self.beliefs['bias_y'] = float(nan)
        self.beliefs['sd_x'] = float(nan)
        self.beliefs['sd_y'] = float(nan)
        self.beliefs.ix['self'] = float(nan)
        self.animation = FuncAnimation(self.figure, self.play, interval=500,
                                       init_func=self.init_plot, blit=True)  # blit=True or False???

    def init_plot(self):
        length = self.field.length
        width = self.field.width
        goal_width = self.field.goal_width
        goal_depth = 1
        self.boundary_plot, = self.game_plot.plot(
            (- length / 2, length / 2, length / 2, - length / 2, - length / 2),
            (- width / 2, - width / 2, width / 2, width / 2, - width / 2), color='gray', linewidth=1)
        self.goal_box_w_plot, = self.game_plot.plot(
            (- length / 2, - length / 2 - goal_depth, - length / 2 - goal_depth, - length / 2),
            (goal_width / 2, goal_width / 2, - goal_width / 2, - goal_width / 2), color='blue', linewidth=1)
        self.goal_box_e_plot, = self.game_plot.plot(
            (length / 2, length / 2 + goal_depth, length / 2 + goal_depth, length / 2),
            (- goal_width / 2, - goal_width / 2, goal_width / 2, goal_width / 2), color='red', linewidth=1)
        self.markings_plot = self.game_plot.scatter(
            self.field.markings_xy.x, self.field.markings_xy.y, color='black', s=24)
        west_x = []
        west_y = []
        east_x = []
        east_y = []
        for i in range(self.num_players):
            if i % 2:
                east_x += [self.players[i].x]
                east_y += [self.players[i].y]
            else:
                west_x += [self.players[i].x]
                west_y += [self.players[i].y]
        self.ball_plot = self.game_plot\
            .scatter(self.ball.x, self.ball.y, s=81, color='magenta', marker='o', animated=True)
        self.west_team_plot = self.game_plot\
            .scatter(west_x, west_y, s=48, c='blue', marker='o', label='West', animated=True)
        self.east_team_plot = self.game_plot\
            .scatter(east_x, east_y, s=48, c='red', marker='o', label='East', animated=True)

        left = - length / 4
        right = length / 4
        top = width / 2 + 1
        self.west_score_text_plot = self.game_plot\
            .text(left, top, self.west_team + ' ' + str(self.score['West']),
                  fontsize=30, color='blue', horizontalalignment='center', animated=True)
        self.east_score_text_plot = self.game_plot\
            .text(right, top, str(self.score['East']) + ' ' + self.east_team,
                  fontsize=30, color='red', horizontalalignment='center', animated=True)

        self.SLAM_plot
        self.cov_plot = imshow(self.players[0].EKF.covariances, interpolation='none', animated=True)

        return self.boundary_plot, self.goal_box_w_plot, self.goal_box_e_plot, self.markings_plot, self.ball_plot,\
            self.west_team_plot, self.east_team_plot, self.west_score_text_plot, self.east_score_text_plot,\
            self.cov_plot

    def play(self, t):
        self.play_per_second()
        west_xy = zeros((ceil(self.num_players / 2), 2))
        east_xy = zeros((floor(self.num_players / 2), 2))
        for i in range(self.num_players):
            if i % 2:
                east_xy[i // 2, 0] = self.players[i].x
                east_xy[i // 2, 1] = self.players[i].y
            else:
                west_xy[i // 2, 0] = self.players[i].x
                west_xy[i // 2, 1] = self.players[i].y
        self.ball_plot.set_offsets(array([[self.ball.x], [self.ball.y]]))
        self.west_team_plot.set_offsets(west_xy)
        self.east_team_plot.set_offsets(east_xy)

        for i in range(self.num_players):
            if self.players[i].lost:
                self.players[i] = Player(x=self.players[i].x, y=self.players[i].y,
                                         velocity=self.players[i].velocity, angle=self.players[i].angle,
                                         acceleration_sigma=self.players[i].acceleration_sigma,
                                         motion_sigma=self.players[i].motion_sigma,
                                         distance_sigma=self.players[i].distance_sigma,
                                         angle_sigma=self.players[i].angle_sigma,
                                         inconsistency_threshold=self.players[i].inconsistency_threshold,
                                         team=self.players[i].team)

        for obj, i in self.players[0].SLAM.items():
            k = 2 * i
            if i:
                marking_x = self.field.markings_xy.ix[obj].x
                marking_y = self.field.markings_xy.ix[obj].y
            xy = self.players[0].EKF.means[k:(k + 2), 0]
            width, height, angle =\
                gaussian_confidence_ellipse_parameters(self.players[0].EKF.covariances[k:(k + 2), k:(k + 2)])
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
                            alpha=0.3, animated=True))

        for obj in (set(self.SLAM_confidence_plots) - set(self.players[0].SLAM)):
            self.SLAM_arrow_plots[obj].set_xdata([0., 0.])
            self.SLAM_arrow_plots[obj].set_ydata([0., 0.])
            self.SLAM_confidence_plots[obj].center = [0., 0.]
            self.SLAM_confidence_plots[obj].width = 0.
            self.SLAM_confidence_plots[obj].height = 0.
            self.SLAM_confidence_plots[obj].angle = 0.

        #self.game_plot.relim()
        #self.game_plot.autoscale_view(True, True, True)

        self.west_score_text_plot.set_text(self.west_team + ' ' + str(self.score['West']))
        self.east_score_text_plot.set_text(str(self.score['East']) + ' ' + self.east_team)

        self.cov_plot.set_data(self.players[0].EKF.covariances)
        self.record_beliefs()

        return [self.ball_plot, self.west_team_plot, self.east_team_plot, self.west_score_text_plot,
                self.east_score_text_plot, self.cov_plot] + list(self.SLAM_confidence_plots.values()) +\
            list(self.SLAM_arrow_plots.values())

    def out_of_play(self):
        return (self.ball.x < -self.field.length / 2) | (self.ball.x > self.field.length / 2) |\
            (self.ball.y < -self.field.width / 2) | (self.ball.y > self.field.width / 2)

    def goal_scored_for(self):
        if self.out_of_play():
            if self.ball.x > self.field.length / 2:
                ball_angle_to_goal_post_se = ray_angle(self.ball.x_just_now, self.ball.y_just_now,
                                                            self.field.markings_xy.ix['GoalPostSE'].x,
                                                            self.field.markings_xy.ix['GoalPostSE'].y)
                ball_angle_to_goal_post_ne = ray_angle(self.ball.x_just_now, self.ball.y_just_now,
                                                            self.field.markings_xy.ix['GoalPostNE'].x,
                                                            self.field.markings_xy.ix['GoalPostNE'].y)
                ball_angle_between_goal_posts_e = angular_difference(ball_angle_to_goal_post_se,
                                                                     ball_angle_to_goal_post_ne)
                ball_angle_between_goal_post_se_and_direction = angular_difference(ball_angle_to_goal_post_se,
                                                                                   self.ball.angle)
                if within_range(ball_angle_between_goal_post_se_and_direction, 0., ball_angle_between_goal_posts_e):
                    return 'West'
            elif self.ball.x < - self.field.length / 2:
                ball_angle_to_goal_post_nw = ray_angle(self.ball.x_just_now, self.ball.y_just_now,
                                                            self.field.markings_xy.ix['GoalPostNW'].x,
                                                            self.field.markings_xy.ix['GoalPostNW'].y)
                ball_angle_to_goal_post_sw = ray_angle(self.ball.x_just_now, self.ball.y_just_now,
                                                            self.field.markings_xy.ix['GoalPostSW'].x,
                                                            self.field.markings_xy.ix['GoalPostSW'].y)
                ball_angle_between_goal_posts_w = angular_difference(ball_angle_to_goal_post_nw,
                                                                     ball_angle_to_goal_post_sw)
                ball_angle_between_goal_post_nw_and_direction = angular_difference(ball_angle_to_goal_post_nw,
                                                                                   self.ball.angle)
                if within_range(ball_angle_between_goal_post_nw_and_direction, 0., ball_angle_between_goal_posts_w):
                    return 'East'

    def play_per_second(self):
        self.time += 1
        index_player_having_ball = None
        min_distance_to_ball = inf
        for i in range(self.num_players):
            self.players[i].orient_toward_ball(self.ball)
            self.players[i].accelerate(normal(scale=self.players[i].acceleration_sigma))
            self.players[i].run()
            self.players[i].observe_marking_in_front(self.field)
            d = self.players[i].distance_to_ball(self.ball)
            if d <= min(min_distance_to_ball, self.players[i].velocity):
                index_player_having_ball = i
                min_distance_to_ball = d

        if index_player_having_ball is not None:
            player = self.players[index_player_having_ball]
            if player.know_goal():
                estimated_goal_x = 0.
                estimated_goal_y = 0.
                known_goal_posts = set(player.target_goal) & set(player.SLAM)
                num_known_goal_posts = len(known_goal_posts)
                for goal_post in known_goal_posts:
                    k = 2 * player.SLAM[goal_post]
                    estimated_goal_x += player.EKF.means[k, 0]
                    estimated_goal_y += player.EKF.means[k + 1, 0]
                estimated_goal_x /= num_known_goal_posts
                estimated_goal_y /= num_known_goal_posts
                estimated_goal_angle = ray_angle(self.ball.x, self.ball.y, estimated_goal_x, estimated_goal_y) +\
                    normal(scale=player.angle_sigma / 2)   # to avoid over-confidence
                self.ball.kicked(self.ball_kick_velocity, estimated_goal_angle)
            elif player.team == 'West':
                self.ball.kicked(self.ball_kick_velocity, normal(scale=player.angle_sigma / 2))  # avoid over-confidence
            elif player.team == 'East':
                self.ball.kicked(self.ball_kick_velocity, pi + normal(scale=player.angle_sigma / 2))  # avoid over-confi

        self.ball.roll()
        if self.out_of_play():
            goal_scored_for = self.goal_scored_for()
            if goal_scored_for:
                self.score[goal_scored_for] += 1
            self.ball = Ball(x=uniform(-self.field.length / 2, self.field.length / 2),
                             y=uniform(-self.field.width / 2, self.field.width / 2),
                             slow_down=self.ball_slow_down)

    def record_beliefs(self):
        x = self.players[0].x
        y = self.players[0].y
        means = self.players[0].EKF.means
        standard_deviations = atleast_2d(self.players[0].EKF.standard_deviations()).T
        for obj, i in self.players[0].SLAM.items():
            k = 2 * i
            if not i:
                self.beliefs.ix[obj].x = x
                self.beliefs.ix[obj].y = y
            self.beliefs.ix[obj].bias_x = means[k, 0] - self.beliefs.ix[obj].x
            self.beliefs.ix[obj].bias_y = means[k + 1, 0] - self.beliefs.ix[obj].y
            self.beliefs.ix[obj].sd_x = standard_deviations[k, 0]
            self.beliefs.ix[obj].sd_y = standard_deviations[k + 1, 0]
        print("TYPICAL PLAYER'S BELIEFS:")
        pprint(self.beliefs)
        print('\n')
