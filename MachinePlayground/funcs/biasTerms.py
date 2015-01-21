from numpy import concatenate, delete, ones


def addBiasElements(arrayA, numsBiases_toAdd = [0, 1]):
    a = arrayA.copy()
    for d in range(len(numsBiases_toAdd)):
        numBiases_toAdd = numsBiases_toAdd[d]
        if numBiases_toAdd > 0:
            s = list(a.shape)
            s[d] = numBiases_toAdd
            a = concatenate((ones(s), a), axis = d)
    return a


def deleteBiasElements(arrayA, numsBiases_toDelete = [1]):
    a = arrayA.copy()
    for d in range(len(numsBiases_toDelete)):
        numBiases_toDelete = numsBiases_toDelete[d]
        if numBiases_toDelete > 0:
            a = delete(a, range(numBiases_toDelete), axis = d)
    return a


def zeroBiasElements(arrayA, numsBiases_toZero_upTo3D = [1]):
    a = arrayA.copy()
    for d in range(len(numsBiases_toZero_upTo3D)):
        numBiases_toZero = numsBiases_toZero_upTo3D[d]
        if numBiases_toZero > 0:
            if d == 0:
                a[range(numBiases_toZero)] = 0
            elif d == 1:
                a[:, range(numBiases_toZero)] = 0
            elif d == 2:
                a[:, :, range(numBiases_toZero)] = 0
    return a