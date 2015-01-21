from legoML.funcs.biasTerms import *


def linearSignals(inputs___matrixCasesInRows, biasesWeights___matrix, inputsAddBiasColumn = True):
    return addBiasElements(inputs___matrixCasesInRows, [0, inputsAddBiasColumn]).dot(biasesWeights___matrix)


def linear_d_over_dInputs_from_d_over_dSignals(d_over_dSignal___matrixCasesInRows, biasesWeights___matrix,
                                               inputsAddBiasColumn = True):
    return d_over_dSignal___matrixCasesInRows.dot(deleteBiasElements(biasesWeights___matrix, [inputsAddBiasColumn]).T)


def linear_d_over_dBiasesWeights_from_d_over_dSignals(d_over_dSignal___matrixCasesInRows, inputs___matrixCasesInRows,
                                                      inputsAddBiasColumn = True):
    return (addBiasElements(inputs___matrixCasesInRows, [0, inputsAddBiasColumn]).T).dot(
        d_over_dSignal___matrixCasesInRows)



def embedNominals_inReals(nominalClassIndices___matrixCasesInRows, reals___matrix):
    m = reals___matrix[]
    return m