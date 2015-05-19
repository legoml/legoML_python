from numpy import array, cos, sin, where, zeros
from math import floor, ceil
import itertools
from dist import *


def subSample(img, sampleEvery = 1):
    return img[::sampleEvery, ::sampleEvery]


def rotate(coords, centerCoords = [0, 0], angle = 0):
    x, y = coords
    x0, y0 = centerCoords
    return [x0 + (x - x0) * cos(angle) + (y - y0) * sin(angle),
            y0 - (x - x0) * sin(angle) + (y - y0) * cos(angle)]


def scale(coords, centerCoords = [0, 0], scales = [1, 1]):
    x, y = coords
    x0, y0 = centerCoords
    scaleX, scaleY = scales
    return [x0 + scaleX * (x - x0), y0 + scaleY * (y - y0)]


def skew(coords, centerCoords = [0, 0], angle = 0):
    x, y = coords
    x0, y0 = centerCoords
    return [x0 + (x - x0) + (y - y0) * tan()]


def pixelInterpol_nearestNeighbor(coords, img):
    x, y = coords
    return img[round(x), round(y)]


def pixelInterpol_linear(coords, img):
    x, y = coords
    x_low = int(x)
    y_low = int(y)
    x_high = x_low + 1
    y_high = y_low + 1
    fracX = x - x_low
    fracY = y - y_low
    return ((1 - fracX) * (1 - fracY) * img[x_low, y_low] + fracX * (1 - fracY) * img[x_high, y_low] +
            (1 - fracX) * fracY * img[x_low, y_high] + fracX * fracY * img[x_high, y_high])


def imgRotate(img, centerCoords, angle, funcInterpolation):

    def yIntercept(x, x0, y0, x1, y1):
        return y0 + (y1 - y0) / (x1 - x0) * (x - x0)

    xMin0 = 0
    yMin0 = 0
    xMax0 = 255
    yMax0 = 255
    x0, y0 = centerCoords
    maxD = ceil(max(distEuclid([xMin0, yMin0], [x0, y0]),
                    distEuclid([xMin0, yMax0], [x0, y0]),
                    distEuclid([xMax0, yMin0], [x0, y0]),
                    distEuclid([xMax0, yMax0], [x0, y0])))
    img1 = zeros([2 * maxD + 1, 2 * maxD + 1])
    img2 = img1.copy()
    xMin1 = maxD - x0
    xMax1 = maxD + (xMax0 - x0) + 1
    yMin1 = maxD - y0
    yMax1 = maxD + (yMax0 - y0) + 1
    img1[xMin1 : xMax1, yMin1 : yMax1] = img
    xXMinYMin, yXMinYMin = rotate([xMin1, yMin1], [maxD, maxD], angle)
    xXMinYMax, yXMinYMax = rotate([xMin1, yMax1], [maxD, maxD], angle)
    xXMaxYMin, yXMaxYMin = rotate([xMax1, yMin1], [maxD, maxD], angle)
    xXMaxYMax, yXMaxYMax = rotate([xMax1, yMax1], [maxD, maxD], angle)
    for i2 in range(2 * maxD + 1):
        yIntercept_XMinYMin_XMinYMax = yIntercept(i2, xXMinYMin, yXMinYMin, xXMinYMax, yXMinYMax)
        yIntercept_XMaxYMin_XMaxYMax = yIntercept(i2, xXMaxYMin, yXMaxYMin, xXMaxYMax, yXMaxYMax)
        set1 = set(range(ceil(min(yIntercept_XMinYMin_XMinYMax, yIntercept_XMaxYMin_XMaxYMax)),
                       floor(max(yIntercept_XMinYMin_XMinYMax, yIntercept_XMaxYMin_XMaxYMax)) + 1))
        yIntercept_XMinYMin_XMaxYMin = yIntercept(i2, xXMinYMin, yXMinYMin, xXMaxYMin, yXMaxYMin)
        yIntercept_XMinYMax_XMaxYMax = yIntercept(i2, xXMinYMax, yXMinYMax, xXMaxYMax, yXMaxYMax)
        set2 = set(range(ceil(min(yIntercept_XMinYMin_XMaxYMin, yIntercept_XMinYMax_XMaxYMax)),
                       floor(max(yIntercept_XMinYMin_XMaxYMin, yIntercept_XMinYMax_XMaxYMax))))
        for j2 in set1.intersection(set2):
            i1, j1 = rotate([i2, j2], [maxD, maxD], -angle)
            img2[i2, j2] = funcInterpolation([i1, j1], img1)
    return img2



def shiftAlongDirection(n):
    d = {0: [0, 1],
         1: [-1, 0],
         2: [0, -1],
         3: [1, 0]}
    return d[n]

def innerBorderIgnoreHole_chainCode(img):
    m = img.max()
    img1 = 1 * (img == m)
    chainCode = []
    firstPixel = where(img1 == 1)
    i0 = firstPixel[0][0]
    j0 = firstPixel[1][0]
    chainCode += [(i0, j0)]
    direction = 3
    i = i0
    j = j0
    di0 = 0
    dj0 = 0
    while (len(chainCode) == 1) or (abs(di0) + abs(dj0) > 0):
        while img1[i + shiftAlongDirection(direction)[0], j + shiftAlongDirection(direction)[1]] == 0:
            direction = (direction + 1) % 4
        chainCode += [direction]
        di, dj = shiftAlongDirection(direction)
        i += di
        j += dj
        di0 += di
        dj0 += dj
        direction = (direction + 3) % 4
    return chainCode


def chainCode_8connectRedundant(chainCode):
    r = zeros(len(chainCode))
    for i in range(1, len(chainCode) - 1):
        mod = (chainCode[i] - chainCode[i + 1]) % 4
        if mod == 1:
            r[i] = 1
    return r



def highlightInnerBorder(img, chainCode, highlightRedundant8Connect = True):
    img1 = img.astype(float)
    m = img1.max()
    r = chainCode_8connectRedundant(chainCode)
    i, j = chainCode[0]
    img1[i, j] = 4 * m
    for item in range(1, len(chainCode)):
        direction = chainCode[item]
        i += shiftAlongDirection(direction)[0]
        j += shiftAlongDirection(direction)[1]
        if highlightRedundant8Connect and (r[item] == 1):
            img1[i, j] = 2 * m
        else:
            img1[i, j] = 4 * m
    return img1



def regionLabel(img):
    m, n = img.shape
    imgLabel = zeros([m, n])
    equivTable = {}
    r = 0
    for i, j in itertools.product(range(m), range(n)):
        if img[i, j] > 0:
            on1 = 1 * ((i > 0) and (j < n - 1) and (img[i - 1, j + 1] > 0))
            if on1:
                lbl1 = imgLabel[i - 1, j + 1]
            else:
                lbl1 = 0
            on2 = 1 * ((i > 0) and (img[i - 1, j] > 0))
            if on2:
                lbl2 = imgLabel[i - 1, j]
            else:
                lbl2 = 0
            on3 = 1 * ((i > 0) and (j > 0) and (img[i - 1, j - 1] > 0))
            if on3:
                lbl3 = imgLabel[i - 1, j - 1]
            else:
                lbl3 = 0
            on4 = 1 * ((j > 0) and (img[i, j - 1] > 0))
            if on4:
                lbl4 = imgLabel[i, j - 1]
            else:
                lbl4 = 0
            if on1 + on2 + on3 + on4 == 0:
                r += 1
                imgLabel[i, j] = r
                equivTable[r] = r
            else:
                a = array([lbl1, lbl2, lbl3, lbl4])
                rMin = a[a > 0].min()
                imgLabel[i, j] = rMin
                if (lbl1 > 0):
                    equivTable[lbl1] = rMin
                if (lbl2 > 0):
                    equivTable[lbl2] = rMin
                if (lbl3 > 0):
                    equivTable[lbl3] = rMin
                if (lbl4 > 0):
                    equivTable[lbl4] = rMin
    print(r)
    for i in range(1, r + 1):
        if (equivTable[i] > i):
            while equivTable[i] > equivTable[equivTable[i]]:
                equivTable[i] = equivTable[equivTable[i]]
    for i, j in itertools.product(range(m), range(n)):
        if imgLabel[i, j] > 0:
            imgLabel[i, j] = equivTable[imgLabel[i, j]]
    return [equivTable, imgLabel]