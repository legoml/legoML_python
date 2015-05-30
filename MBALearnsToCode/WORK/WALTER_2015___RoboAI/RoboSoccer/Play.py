from numpy import pi
from MBALearnsToCode.WORK.WALTER_2015___RoboAI.RoboSoccer.RoboSoccer import Game


num_players_per_team = 1
velocity = 6.
motion_sigma = .3
distance_sigma = 1.
angle_sigma = 3 * pi / 180

ball_kick_velocity = 9.
ball_slow_down = .8

Game(num_players_per_team=num_players_per_team,
     velocity=velocity,
     motion_sigma=motion_sigma,
     distance_sigma=distance_sigma,
     angle_sigma=angle_sigma,
     ball_kick_velocity=ball_kick_velocity,
     ball_slow_down=ball_slow_down)
