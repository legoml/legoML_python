from numpy import amax, abs
from scipy.spatial.distance import euclidean, cityblock


def distEuclid(v1, v2):
    return euclidean(v1, v2)


def distCityBlock(v1, v2):
    return cityblock(v1, v2)


def distChessboard(v1, v2):
    return amax(abs(v1 - v2))
