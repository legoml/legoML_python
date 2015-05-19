from numpy import *
from PIL import Image
from zzzAssignments import *
from imageFuncs import *
from matplotlib.pyplot import *
import skimage.io as skimageIO
import skimage.measure as skimageMeasure



# HOMEWORK 1

# Assignment 1(i) - contours with Euclidean distances
assignment0101(i0 = 15, j0 = 15, dists = [6, 9, 12], func_dist = distEuclid)
input('Press <ENTER> to continue')

# Assignment 1(ii) - contours with CityBlock distances
assignment0101(i0 = 15, j0 = 15, dists = [6, 9, 12], func_dist = distCityBlock)
input('Press <ENTER> to continue')

# Assignment 1(iii) - contours with Chessboard distances
assignment0101(i0 = 15, j0 = 15, dists = [6, 9, 12], func_dist = distChessboard)
input('Press <ENTER> to continue')

# Assignment 2(i) - colours increasing with Euclidean distances
assignment0102i(i0 = 15, j0 = 15, func_dist = distEuclid)
input('Press <ENTER> to continue')

# Assignment 2(ii) - colours increasing with CityBlock distances
assignment0102i(i0 = 15, j0 = 15, func_dist = distCityBlock)
input('Press <ENTER> to continue')

# Assignment 2(iii) - colours increasing with Chessboard distances
assignment0102i(i0 = 15, j0 = 15, func_dist = distChessboard)
input('Press <ENTER> to continue')

# Assignment 2(iv) - region with Euclidean distance < 10
assignment0102i(i0 = 15, j0 = 15, func_dist = distEuclid, maxD = 10)
input('Press <ENTER> to continue')

# Assignment 2(v) - run-length code of region with Euclidean distance < 10
runLengthCode = assignment0102ii(i0 = 15, j0 = 15, func_dist = distEuclid, maxD = 10)
input('Press <ENTER> to continue')

# Assignment 3(i) - reconstruction using Run-Length Code
img = assignment0103i(runLengthCode)
input('Press <ENTER> to continue')

# Assignment 3(ii) - sub-sample
imgSubSampled = imshow(subSample(img, 4), cmap = cm.Greys_r, interpolation = 'none')
input('Press <ENTER> to continue')



# HOMEWORK 2

# Assignment 1(i) - nearest-neighbour interpolation
img = array(PIL.Image.open('figure_problem_set_2.tiff'))
img2i = imgRotate(img, [128, 128], -math.pi / 6, pixelInterpol_nearestNeighbor)
imshow(img2i, cmap = cm.Greys_r, interpolation = 'none')
input('Press <ENTER> to continue')
# Assignment 1(ii) - linear interpolation
img2ii = imgRotate(img, [128, 128], -math.pi / 6, pixelInterpol_linear)
imshow(img2ii, cmap = cm.Greys_r, interpolation = 'none')
input('Press <ENTER> to continue')

# Assignment 2 - gradients via Sobel operator
img = array(PIL.Image.open('figure_problem_set_2.tiff'))
h1 = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
imgH1 = convolve2d(img, h1)
imshow(imgH1, cmap = cm.Greys_r, interpolation = 'none')
h2 = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
imgH2 = convolve2d(img, h2)
imshow(imgH2, cmap = cm.Greys_r, interpolation = 'none')
gradientAngle = arctan2(h1, h2)
imshow(gradientAngle, cmap = cm.Greys_r, interpolation = 'none')



# HOMEWORK 3

# Assignment 1(i) - segment image and highlight border
img = array(Image.open('figure1_problem_set_3.tiff'))
chainCode = innerBorderIgnoreHole_chainCode(img)
img1 = highlightInnerBorder(img, chainCode, False)
imshow(img1, cmap = cm.Greys_r, interpolation = 'none')

# Assignment 1(ii) - highlight border using 8-connectivity
img2 = highlightInnerBorder(img, chainCode, True)
imshow(img2, cmap = cm.Greys_r, interpolation = 'none')

# Assignment 1(iii) - compute 4-connected and 8-connected perimeter
perimeter_4Connect = len(chainCode) - 1
perimeter_4Connect
perimeter_8Connect = len(chainCode) - 1 - sum(chainCode_8connectRedundant(chainCode))
perimeter_8Connect



# Assignment 2 - show that border-tracing algorithm does not work with different
# initial point
# (repeat assignment 1(i) with code for the "chainCode" function distorted



# Assignment 3 - region labeling
img = array(Image.open('figure2_problem_set_3.tiff'))
equivTable, imgLabel = regionLabel(img)
print(equivTable)
imshow(imgLabel, cmap = cm.Greys_r, interpolation = 'none')


# Assignment 4 - Morphology # - not done yet
img = array(Image.open('figure_problem_set_4.tiff'))
img = 1 * (img > 1)
imshow(img, cmap = cm.Greys_r, interpolation = 'none')
equivTable, imgLabel = regionLabel(img)
imshow(imgLabel, cmap = cm.Greys_r, interpolation = 'none')


# HOMEWORK 6
# Assignment 3 - calculate moments
img = skimageIO.imread('img_moment.tif').astype(float)
imshow(img, cmap = cm.Greys_r, interpolation = 'none')
moments_deg1 = skimageMeasure.moments(img, order = 1)
x0 = moments_deg1[0, 1] / moments_deg1[0, 0]
y0 = moments_deg1[1, 0] / moments_deg1[0, 0]
momentsCentral_deg2 = skimageMeasure.moments_central(img, x0, y0, order = 2)
m00 = momentsCentral_deg2[0, 0]
m11 = momentsCentral_deg2[1, 1]
m02 = momentsCentral_deg2[0, 2]
m20 = momentsCentral_deg2[2, 0]
orientation = degrees(arctan2(2 * m11, (m20 - m02)) / 2)

# HOUGH TRANSFORM HOMEWORK
img = skimageIO.imread('img_hough_circle.tiff').astype(float)

# Enhance edges by Sobel operator
h1 = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
imgH1 = convolve2d(img, h1)
h2 = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
imgH2 = convolve2d(img, h2)
img = 1.0 * (sqrt(imgH1 ** 2 + imgH2 ** 2) > 0)

m, n = img.shape
nonZeroIs, nonZeroJs = nonzero(img)
imgCircles = zeros([m, n])
houghImages = {}

for r in range(25, 70):
    print(r)
    houghImages[r] = zeros([m, n])
    for idx in range(len(nonZeroIs)):
        i = nonZeroIs[idx]
        j = nonZeroJs[idx]
        for dI in range(r + 1):
            dJ = round(sqrt(r ** 2 - dI ** 2))
            if (0 <= (i - dI) < m) and (0 <= (j - dJ) < n):
                houghImages[r][i - dI, j - dJ] += 1
            if (0 <= (i - dI) < m) and (0 <= (j + dJ) < n):
                houghImages[r][i - dI, j + dJ] += 1
            if (0 <= (i + dI) < m) and (0 <= (j - dJ) < n):
                houghImages[r][i + dI, j - dJ] += 1
            if (0 <= (i + dI) < m) and (0 <= (j + dJ) < n):
                houghImages[r][i + dI, j + dJ] += 1
    houghImages[r] = (houghImages[r] >= 3 * r) # detect majority of 2 * pi * r points on a circle
    nonZeroKs, nonZeroLs = nonzero(houghImages[r])
    for idx in range(len(nonZeroKs)):
        k = nonZeroKs[idx]
        l = nonZeroLs[idx]
        for dK in range(r + 1):
            dL_f = floor(sqrt(r ** 2 - dK ** 2))
            dL_c = ceil(sqrt(r ** 2 - dK ** 2))
            img[k - dK, (l - dL_c) : (l - dL_f)] = 3
            img[k - dK, (l + dL_f) : (l + dL_c)] = 3
            img[k + dK, (l - dL_c) : (l - dL_f)] = 3
            img[k + dK, (l + dL_f) : (l + dL_c)] = 3

imshow(img, cmap = cm.Greys_r, interpolation = 'none')



### PCA Exercise
imgs = zeros([32, 128, 128])
for i in range(32):
    imgs[i] = skimageIO.imread('biq' + str(i + 1).zfill(3) + '.tif').astype(float)
imgs = imgs.reshape([32, 128 * 128])

mu = mean(imgs, 0)
sigma = std(imgs, 0)
imgs = (imgs - mu) / sigma # normalize data

covMatrix_ofTranspose = imgs.dot(imgs.T) / 32
eigenvalues, eigenvectors = linalg.eigh(covMatrix_ofTranspose)
topEigenvectors = eigenvectors[:, 31:23:(-1)]

originalEigenvectors = (imgs.T).dot(topEigenvectors)
for i in range(8):
    originalEigenvectors[:, i] /= distEuclid(originalEigenvectors[:, i], 0)

eigenFaces = ((originalEigenvectors.T)).reshape([8, 128, 128])
imshow(eigenFaces[0].reshape([128, 128]),
       cmap = cm.Greys_r, interpolation = 'none')

firstFace = imgs[0]
firstFace_reconstructed = zeros(firstFace.shape)
num_components = 8
for i in range(num_components):
    originalEigenvector = originalEigenvectors[:, i]
    firstFace_reconstructed += firstFace.dot(originalEigenvector) * originalEigenvector
imshow((mu + sigma * firstFace_reconstructed).reshape([128, 128]),
       cmap = cm.Greys_r, interpolation = 'none')