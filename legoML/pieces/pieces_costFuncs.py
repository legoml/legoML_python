def pc_squareError_half():
    return lambda Arr1, Arr2: sqErr(Arr1, Arr2, False)


def pcSquareErrorAvg():
    return lambda Arr1, Arr2: sqErr(Arr1, Arr2, True)


