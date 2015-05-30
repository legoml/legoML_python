from __future__ import print_function, division
from copy import deepcopy
from pprint import pprint
from numpy import abs, allclose, arctan2, array, atleast_2d, cos,  degrees, eye, hstack, inf, nan, pi, sin, sqrt,\
    vstack, zeros
from numpy.linalg import eigh
from numpy.random import normal, uniform, random
from pandas import DataFrame
from matplotlib.pyplot import figure, subplot, imshow
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from MBALearnsToCode.Classes.CLASSES___KalmanFilters import ExtendedKalmanFilter as EKF
from MBALearnsToCode.Functions.FUNCTIONS___zzzMISC import approx_gradients


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
    def __init__(self, x=0., y=0., velocity=0., angle=0., motion_sigma=0., distance_sigma=0., angle_sigma=0.,
                 team='West'):
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

    def transition_means(self, all_xy___vector, velocity_and_angle):
        conditional_xy_means___vector = deepcopy(all_xy___vector)
        v, a = velocity_and_angle
        conditional_xy_means___vector[0, 0] += v * cos(a)
        conditional_xy_means___vector[1, 0] += v * sin(a)
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

    def observation_means(self, all_xy___vector, marking_indices=tuple()):
        n_markings = len(marking_indices)
        conditional_observation_means___vector = zeros((2 * n_markings, 1))
        self_x, self_y = all_xy___vector[0:2, 0]
        for i in range(n_markings):
            k = 2 * i
            k_marking = 2 * marking_indices[i]
            marking_x, marking_y = all_xy___vector[k_marking:(k_marking + 2), 0]
            conditional_observation_means___vector[k, 0] = euclidean_distance(self_x, self_y, marking_x, marking_y)
            conditional_observation_means___vector[k + 1, 0] = relative_angle(self_x, self_y, marking_x, marking_y)
        return conditional_observation_means___vector

    def observation_means_jacobi(self, all_xy___vector, marking_indices=tuple()):
        n_markings = len(marking_indices)
        conditional_observation_means_jacobi___matrix = zeros((2 * n_markings, all_xy___vector.size))
        self_x, self_y = all_xy___vector[0:2, 0]
        for i in range(n_markings):
            k = 2 * i
            k_marking = 2 * marking_indices[i]
            marking_x, marking_y = all_xy___vector[k_marking:(k_marking + 2), 0]
            conditional_observation_means_jacobi___matrix[k, (0, 1, k_marking, k_marking + 1)] =\
                euclidean_distance_gradients(self_x, self_y, marking_x, marking_y)
            conditional_observation_means_jacobi___matrix[k + 1, (0, 1, k_marking, k_marking + 1)] =\
                relative_angle_gradients(self_x, self_y, marking_x, marking_y)
        return conditional_observation_means_jacobi___matrix

    def observation_covariances(self, observations___vector, marking_indices=tuple()):
        n_markings = len(marking_indices)
        conditional_observation_covariances___matrix = zeros((2 * n_markings, 2 * n_markings))
        for i in range(n_markings):
            k = 2 * i
            conditional_observation_covariances___matrix[k, k] = self.distance_sigma ** 2
            conditional_observation_covariances___matrix[k + 1, k + 1] = self.angle_sigma ** 2
        return conditional_observation_covariances___matrix

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
        observed_angle = relative_angle(self.x, self.y, marking.x, marking.y) +\
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

    def observe(self, markings, min_distance_over_distance_sigma_ratio=3.):
        observations___vector = array([[]]).T
        marking_indices = []
        for marking in markings:
            if marking.id in self.SLAM:
                observed_distance = euclidean_distance(self.x, self.y, marking.x, marking.y) +\
                    normal(scale=self.distance_sigma / 2)  # to avoid over-confidence
                if observed_distance >= min_distance_over_distance_sigma_ratio * self.distance_sigma:
                    observed_angle = relative_angle(self.x, self.y, marking.x, marking.y) +\
                        normal(scale=self.angle_sigma / 2)   # to avoid over-confidence
                    observations___vector = vstack((observations___vector,
                                                    observed_distance,
                                                    observed_angle))
                    marking_indices += [self.SLAM[marking.id]]
        if marking_indices:
            self.EKF.update(observations___vector, marking_indices=marking_indices)
        for marking in markings:
            if marking.id not in self.SLAM:
                self.augment_map(marking)

    def observe_marking_in_front(self, field, min_distance_over_distance_sigma_ratio=3.):
        min_angle = pi
        marking_in_front = None
        for marking in field.markings:
            a = abs(angular_difference(self.angle, relative_angle(self.x, self.y, marking.x, marking.y)))
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
    def __init__(self, num_players_per_team=11, velocity=6., motion_sigma=0.3, distance_sigma=1.,
                 angle_sigma=3 * pi / 180, ball_kick_velocity=3., ball_slow_down=0.8):
        self.field = Field()
        length = self.field.length
        width = self.field.width
        self.num_players_per_team = num_players_per_team
        self.players = []
        for i in range(num_players_per_team):
            self.players += [Player(uniform(- length / 2, 0), uniform(- width / 2, width / 2),
                                    velocity, uniform(-pi, pi), motion_sigma, distance_sigma, angle_sigma, 'West'),
                             Player(uniform(0, length / 2), uniform(- width / 2, width / 2),
                                    velocity, uniform(-pi, pi), motion_sigma, distance_sigma, angle_sigma, 'East')]

        self.ball_kick_velocity = ball_kick_velocity
        self.ball_slow_down = ball_slow_down
        self.ball = Ball(slow_down=ball_slow_down)

        self.time = 0

        self.figure = figure()
        gs = GridSpec(1, 2, width_ratios=[5, 1])
        self.game_plot = subplot(gs[0])
        self.SLAM_plot = subplot(gs[1])
        self.boundary_plot = None
        self.markings_plot = None
        self.ball_plot = None
        self.west_team_plot = None
        self.east_team_plot = None
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
                                       init_func=self.init_plot, blit=True)

    def init_plot(self):
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
            .scatter(self.ball.x, self.ball.y, s=36, color='magenta', marker='o', animated=True)
        self.west_team_plot = self.game_plot\
            .scatter(west_x, west_y, s=48, c='blue', marker='o', label='West', animated=True)
        self.east_team_plot = self.game_plot\
            .scatter(east_x, east_y, s=48, c='red', marker='o', label='East', animated=True)
        self.SLAM_plot
        self.cov_plot = imshow(self.players[0].EKF.covariances, interpolation='none', animated=True)
        return self.boundary_plot, self.markings_plot, self.ball_plot, self.west_team_plot, self.east_team_plot,\
            self.cov_plot

    def play(self, t):
        self.play_per_second()
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
        self.cov_plot.set_data(self.players[0].EKF.covariances)
        self.record_beliefs()
        return [self.ball_plot, self.west_team_plot, self.east_team_plot, self.cov_plot] +\
            list(self.SLAM_confidence_plots.values()) + list(self.SLAM_arrow_plots.values())

    def out_of_play(self):
        return (self.ball.x < -self.field.length / 2) | (self.ball.x > self.field.length / 2) |\
            (self.ball.y < -self.field.width / 2) | (self.ball.y > self.field.width / 2)

    def goal_scored(self):
        return True

    def play_per_second(self):
        self.time += 1
        self.ball.roll()
        player_having_ball_and_knowing_goal = None
        min_distance_to_ball = inf
        for i in range(0, 2 * self.num_players_per_team):
            self.players[i].orient_toward_ball(self.ball)
            self.players[i].run()
            self.players[i].observe_marking_in_front(self.field)
            d = self.players[i].distance_to_ball(self.ball)
            if d < min(min_distance_to_ball, self.players[i].velocity) and self.players[i].know_goal():
                player_having_ball_and_knowing_goal = i
                min_distance_to_ball = d

        if player_having_ball_and_knowing_goal:
            player = self.players[player_having_ball_and_knowing_goal]
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
            estimated_goal_angle = relative_angle(self.ball.x, self.ball.y, estimated_goal_x, estimated_goal_y)
            self.ball.kicked(self.ball_kick_velocity, estimated_goal_angle)

        if self.out_of_play():
            self.ball = Ball(slow_down=self.ball_slow_down)

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
        print("\nTYPICAL PLAYER'S BELIEFS:")
        pprint(self.beliefs)
        print('\n')


def confidence_ellipse_parameters(cov, nstd=2):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    angle = degrees(arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * sqrt(vals)

    return width, height, angle


def UNIT_TEST___WALTER_2015___RoboAI___RoboSoccer___FunctionGradients(num_times=1000):
    num_distance_successes = 0
    num_angle_successes = 0
    for t in range(num_times):
        vector = 1000 * random(4)
        distance_gradients___analytic = array(euclidean_distance_gradients(*vector))
        distance_gradients___approx = approx_gradients(lambda v: euclidean_distance(*v), vector)
        num_distance_successes += allclose(distance_gradients___approx, distance_gradients___analytic)
        angle_gradients___analytic = array(relative_angle_gradients(*vector))
        angle_gradients___approx = approx_gradients(lambda v: relative_angle(*v), vector)
        num_angle_successes += allclose(angle_gradients___approx, angle_gradients___analytic)
    print(distance_gradients___analytic)
    print(distance_gradients___approx)
    print(angle_gradients___analytic)
    print(angle_gradients___approx)
    return 100 * num_angle_successes / num_times, 100 * num_angle_successes / num_times