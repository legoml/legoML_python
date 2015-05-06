from copy import deepcopy
from numpy import delete, zeros
import itertools


def matrix_minor(matrix, rows_to_delete=None, columns_to_delete=None):
    m = deepcopy(matrix)
    if rows_to_delete is not None:
        m = delete(m, rows_to_delete, axis=0)
    if columns_to_delete is not None:
        m = delete(m, columns_to_delete, axis=1)
    return m


def sympy_matrix_cofactor_expansion(matrix, row_num, column_num):
    return ((-1) ** (row_num + column_num)) *\
        sympy_matrix_determinant_expansion(matrix_minor(matrix, row_num, column_num))


def sympy_matrix_determinant_expansion(matrix_of_symbols):
    if matrix_of_symbols.size == 1:
        return matrix_of_symbols[0, 0]
    else:
        d = 0
        for i in range(matrix_of_symbols.shape[0]):
            d += matrix_of_symbols[i, 0] * sympy_matrix_cofactor_expansion(matrix_of_symbols, i, 0)
    return d


def sympy_matrix_inverse_expansion(matrix_of_symbols):
    n = matrix_of_symbols.shape[0]
    cofactors = zeros([n, n]).astype(object)
    r = range(n)
    for i, j in itertools.product(r, r):
        cofactors[i, j] = sympy_matrix_cofactor_expansion(matrix_of_symbols, i, j)
    return cofactors / sympy_matrix_determinant_expansion(matrix_of_symbols)