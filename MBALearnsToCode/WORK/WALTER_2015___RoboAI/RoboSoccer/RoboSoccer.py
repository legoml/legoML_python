from numpy import array
from matplotlib.pyplot import imshow, subplots

from skimage import io

from PIL import Image
from zzzAssignments import *
from imageFuncs import *
from matplotlib.pyplot import *
import skimage.io as skimageIO
import skimage.measure as skimageMeasure

class Field:
    def __init__(self, length, width):
        self.center = array([[0.], [0.]])
        self.corner_sw = array([[]])


class Player:
    def __init__(self, xy):
        self.xy = xy