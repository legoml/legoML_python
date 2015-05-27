from MBALearnsToCode.WORK.WALTER_2015___RoboAI.RoboSoccer.RoboSoccer import Game
from MBALearnsToCode.WORK.WALTER_2015___RoboAI.RoboSoccer.ConfidenceEllipses import confidence_ellipse_parameters
from matplotlib.patches import Ellipse
from matplotlib.pyplot import figure, subplots, show
from numpy import array

#g = Game(1)
#g.players_run_randomly(100)


pos = [0., 0.]
cov = array([[1., 1.5], [1.5, 4.]])
width, height, angle = confidence_ellipse_parameters(cov)
print(width, height, angle)

fig = figure()
ax = fig.add_subplot(111, aspect='equal')
ax.add_artist(Ellipse(pos, width, height, angle))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

from pylab import figure, show, rand
from matplotlib.patches import Ellipse

NUM = 250

ells = [Ellipse(xy=rand(2)*10, width=rand(), height=rand(), angle=rand()*360)
        for i in range(NUM)]

fig = figure()
ax = fig.add_subplot(111, aspect='equal')
for e in ells:
    ax.add_artist(e)
    #e.set_clip_box(ax.bbox)
   # e.set_alpha(rand())
   # e.set_facecolor(rand(3))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

show()