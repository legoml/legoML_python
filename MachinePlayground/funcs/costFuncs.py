# COSTFUNCS module defines common cost functions such as Squared Error (for Linear Regression)
# and Cross Entropy (for Logistic and Softmax Regressions)



from numpy import divide, multiply, power



def squareError_half(hypoArray, targetArray, avg = False, *args, **kwargs):
    f = power(hypoArray - targetArray, 2).sum() / 2
    if avg:
        numCases = max(arr1.shape[1], arr2.shape[1])
        return f / numCases
    else:
        return f


def rootMeanSquareError(hypoArray, targetArray, avg = False):
    return sqrt(2 * squareError_half(arr1, arr2, avg = True))


def crossEntropy(ofArr, fromArr, classSkewnesses = 1, tiny = 1e-18):
    return - divide(multiply(fromArr, log(ofArr + tiny)), classSkewnesses).sum()



def crossEntropy_binaryClasses(hypoArr, targetArr, positiveSkewnesses = 1, avg = False):
    negativeSkewnesses = 2 - positiveSkewnesses
    f = (crossEntropy(hypoArr, targetArr, positiveSkewnesses) +
        crossEntropy(1 - hypoArr, 1 - targetArr, negativeSkewnesses))
    if avg:
        numCases = max(arr1.shape[1], arr2.shape[1])
        return f / numCases
    else:
        return f



def crossEntropy_multiClasses(hypoRowMat, targetRowMat, skewnesses = 1, avg = False):

    if avg:
        numCases = max(arr1.shape[1], arr2.shape[1])
        return f / numCases
    else:
        return f


