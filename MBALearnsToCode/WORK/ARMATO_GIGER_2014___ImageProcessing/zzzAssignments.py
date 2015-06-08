from numpy import *
from scipy.signal import convolve2d
from matplotlib.pyplot import *
from PIL import *
import itertools
from dist import *
from imageFuncs import *


def assignment0101(i0 = 15, j0 = 15, dists = [10], func_dist = distEuclid, dimSize = 32):
    img = zeros([dimSize, dimSize]).astype(uint8)
    maxD = max(dists)
    minI = max(i0 - maxD, 0)
    maxI = min(i0 + maxD + 1, dimSize)
    minJ = max(j0 - maxD, 0)
    maxJ = min(j0 + maxD + 1, dimSize)
    for i, j, d in itertools.product(range(minI, maxI), range(minJ, maxJ), dists):
        if round(func_dist(matrix([i, j]), matrix([i0, j0])) - d) == 0:
            img[i, j] = 255
    imshow(img, cmap = cm.Greys_r, interpolation = 'none')
    return img


def assignment0102i(i0 = 15, j0 = 15, func_dist = distEuclid, dimSize = 32, maxD = 32):
    img = zeros([dimSize, dimSize]).astype(uint8)
    minI = max(i0 - maxD, 0)
    maxI = min(i0 + maxD + 1, dimSize)
    minJ = max(j0 - maxD, 0)
    maxJ = min(j0 + maxD + 1, dimSize)
    for i, j in itertools.product(range(minI, maxI), range(minJ, maxJ)):
        d = round(func_dist(matrix([i, j]), matrix([i0, j0])))
        if d <= maxD:
            img[i, j] = d
    imshow(img, cmap = cm.Greys_r, interpolation = 'none')
    return img



def assignment0102ii(i0 = 15, j0 = 15, func_dist = distEuclid, dimSize = 32, maxD = 32):
    img = zeros([dimSize, dimSize]).astype(uint8)
    minI = max(i0 - maxD, 0)
    maxI = min(i0 + maxD + 1, dimSize)
    minJ = max(j0 - maxD, 0)
    maxJ = min(j0 + maxD + 1, dimSize)
    for i, j in itertools.product(range(minI, maxI), range(minJ, maxJ)):
        if 0.0 < round(func_dist(matrix([i, j]), matrix([i0, j0]))) <= maxD:
            img[i, j] = 255
    imshow(img, cmap = cm.Greys_r, interpolation = 'none')
    runLengthCode = []
    for i in range(dimSize):
        if img[i, :].sum() > 0:
            j = 0
            while j < dimSize:
                while (j < dimSize) and (img[i, j] == 0):
                    j += 1
                if (j < dimSize) and (img[i, j] > 0):
                    startJ = j
                    while (j < dimSize) and (img[i, j] > 0):
                        j += 1
                    endJ = j
                    runLengthCode.append([i, startJ, endJ])
    return runLengthCode


def assignment0103i(runLengthCode, dimSize = 32):
    img = zeros([dimSize, dimSize]).astype(uint8)
    for l in runLengthCode:
        i, startJ, endJ = l
        img[i, range(startJ, endJ)] = 255
    imshow(img, cmap = cm.Greys_r, interpolation = 'none')
    return img


