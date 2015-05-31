from numpy import pi
from MBALearnsToCode.WORK.WALTER_2015___RoboAI.RoboSoccer.RoboSoccer import Game


num_players = 6
velocity = 6.
acceleration_sigma = 1.
motion_sigma = .1
distance_sigma = 1.
angle_sigma = 3 * pi / 180
inconsistency_threshold = 6.

ball_kick_velocity = 8.
ball_slow_down = .68

team_names = ('Juventus', 'Barcelona')

Game(num_players=num_players,
     velocity=velocity,
     acceleration_sigma=acceleration_sigma,
     motion_sigma=motion_sigma,
     distance_sigma=distance_sigma,
     angle_sigma=angle_sigma,
     inconsistency_threshold=inconsistency_threshold,
     ball_kick_velocity=ball_kick_velocity,
     ball_slow_down=ball_slow_down,
     team_names=team_names)
