from numpy import exp, log, sqrt

from MBALearnsToCode import Piece


def PIECE___average_half_square_error():
    """PIECE___average_half_squared_error

    AVERAGE_HALF_SQUARE_ERROR = 1/2 x sum of squared differences between items in PREDICTED_OUTPUTS multi-dimensional
    array and TARGET_OUTPUTS multi-dimensional array, divided by the number of cases
    """

    forwards = {'average_half_square_error':
                    [lambda a0, a1: ((a1 - a0) ** 2).sum() / (2 * a0.shape[0]),
                     {'a0': 'target_outputs',
                      'a1': 'predicted_outputs'}]}

    backwards = {('DOVERD', 'average_half_square_error', 'predicted_outputs'):
                     [lambda a0, a1: (a1 - a0) / a0.shape[0],
                      {'a0': 'target_outputs',
                       'a1': 'predicted_outputs'}]}

    return Piece(forwards, backwards)



def PIECE___root_mean_square_error():
    """PIECE___root_mean_square_error:

    ROOT_MEAN_SQUARE_ERROR = sum of squared differences between items in PREDICTED_OUTPUTS multi-dimensional
    array and TARGET_OUTPUTS multi-dimensional array, divided by the number of cases, then square-rooted
    """

    forwards = {'root_mean_square_error':
                    [lambda a0, a1: sqrt(((a1 - a0) ** 2).sum() / a0.shape[0]),
                     {'a0': 'target_outputs',
                      'a1': 'predicted_outputs'}]}

    backwards = {('DOVERD', 'root_mean_square_error', 'predicted_outputs'):
                     [lambda a0, a1, rmse: ((a1 - a0) / a0.shape[0]) / rmse,
                      {'a0': 'target_outputs',
                       'a1': 'predicted_outputs',
                       'rmse': 'root_mean_square_error'}]}

    return Piece(forwards, backwards)



def PIECE___root_mean_square_error_from_average_half_square_error():
    """PIECE___root_mean_square_error_from_average_half_square_error:

    ROOT_MEAN_SQUARE_ERROR = square root of 2 x AVERAGE_HALF_SQUARE_ERROR
    """

    forwards = {'root_mean_square_error':
                    [lambda avg_half_se: sqrt(2 * avg_half_se),
                     {'avg_half_se': 'average_half_square_error'}]}

    backwards = {('DOVERD', 'root_mean_square_error', 'average_half_square_error'):
                     [lambda rmse: 1. / rmse,
                      {'rmse': 'root_mean_square_error'}]}

    return Piece(forwards, backwards)



def PIECE___average_binary_class_cross_entropy():
    """PIECE___average_binary_class_cross_entropy:

    AVERAGE_BINARY_CLASS_CROSS_ENTROPY = cross entropy from TARGET_OUTPUTS multi-dimensional array (of numbers
    between 0 and 1, inclusive) of PREDICTED_OUTPUTS multi-dimensional array (of numbers between 0 and 1, inclusive),
    divided by the number of cases
    """

    forwards = {'average_binary_class_cross_entropy':
                    [lambda from_arr, of_arr, tiny = exp(-36):
                        - ((from_arr * log(of_arr + tiny))
                        + ((1. - from_arr) * log(1. - of_arr + tiny))).sum() / from_arr.shape[0],
                    {'from_arr': 'target_outputs',
                     'of_arr': 'predicted_outputs'}]}

    backwards = {('DOVERD', 'average_binary_class_cross_entropy', 'predicted_outputs'):
                    [lambda from_arr, of_arr:
                        - ((from_arr / of_arr)
                        - ((1. - from_arr) / (1. - of_arr))) / from_arr.shape[0],
                     {'from_arr': 'target_outputs',
                      'of_arr': 'predicted_outputs'}]}

    return Piece(forwards, backwards)



def PIECE___average_unskewed_binary_class_cross_entropy():
    """PIECE___average_unskewed_binary_class_cross_entropy:

    AVERAGE_UNSKEWED_BINARY_CLASS_CROSS_ENTROPY = cross entropy from TARGET_OUTPUTS multi-dimensional array (of numbers
    between 0 and 1, inclusive) of PREDICTED_OUTPUTS multi-dimensional array (of numbers between 0 and 1, inclusive),
    divided by the number of cases, adjusted by skewnesses between the positive (1) and negative (0) classes that are
    represented by POSITIVE_CLASS_SKEWNESSES multi-dimensional array
    """

    forwards = {'average_unskewed_binary_class_cross_entropy':
                    [lambda from_arr, of_arr, pos_skew, tiny = exp(-36):
                        - ((from_arr * log(of_arr + tiny)) / pos_skew
                        + ((1. - from_arr) * log(1. - of_arr + tiny)) / (2. - pos_skew)).sum() / from_arr.shape[0],
                    {'from_arr': 'target_outputs',
                     'of_arr': 'predicted_outputs',
                     'pos_skew': 'positive_class_skewnesses'}]}

    backwards = {('DOVERD', 'average_unskewed_binary_class_cross_entropy', 'predicted_outputs'):
                    [lambda from_arr, of_arr, pos_skew:
                        - ((from_arr / of_arr) / pos_skew
                        - ((1. - from_arr) / (1. - of_arr)) / (2. - pos_skew)) / from_arr.shape[0],
                     {'from_arr': 'target_outputs',
                      'of_arr': 'predicted_outputs',
                      'pos_skew': 'positive_class_skewnesses'}]}

    return Piece(forwards, backwards)



def PIECE___average_multi_class_cross_entropy():
    """PIECE___average_multi_class_cross_entropy:

    AVERAGE_MULTI_CLASS_CROSS_ENTROPY = cross entropy from TARGET_OUTPUTS matrix (of numbers between 0 and 1,
    inclusive, with cases in rows) of PREDICTED_OUTPUTS matrix (of numbers between 0 and 1, inclusive, with cases in
    rows), divided by the number of cases
    """

    forwards = {'average_multi_class_cross_entropy':
                    [lambda from_mat, of_mat, tiny = exp(-36):
                        - (from_mat * log(of_mat + tiny)).sum() / from_mat.shape[0],
                     {'from_mat': 'target_outputs',
                      'of_mat': 'predicted_outputs'}]}

    backwards = {('DOVERD', 'average_multi_class_cross_entropy', 'predicted_outputs'):
                     [lambda from_mat, of_mat:
                        - (from_mat / of_mat) / from_mat.shape[0],
                      {'from_mat': 'target_outputs',
                       'of_mat': 'predicted_outputs'}]}

    return Piece(forwards, backwards)



def PIECE___average_unskewed_multi_class_cross_entropy():
    """PIECE___average_unskewed_multi_class_cross_entropy:

    AVERAGE_UNSKEWED_MULTI_CLASS_CROSS_ENTROPY = cross entropy from TARGET_OUTPUTS matrix (of numbers between 0 and 1,
    inclusive, with cases in rows) of PREDICTED_OUTPUTS matrix (of numbers between 0 and 1, inclusive, with cases in
    rows), divided by the number of cases, adjusted by skewnesses among the classes that are represented by
    CLASS_SKEWNESSES matrix
    """

    forwards = {'average_unskewed_multi_class_cross_entropy':
                    [lambda from_mat, of_mat, skew, tiny = exp(-36):
                        - ((from_mat * log(of_mat + tiny))
                           / ((from_mat.shape[1] / skew.sum(1, keepdims = True)) * skew)).sum() / from_mat.shape[0],
                     {'from_mat': 'target_outputs',
                      'of_mat': 'predicted_outputs',
                      'skew': 'multi_class_skewnesses'}]}

    backwards = {('DOVERD', 'average_unskewed_multi_class_cross_entropy', 'predicted_outputs'):
                     [lambda from_mat, of_mat, skew:
                        - ((from_mat / of_mat)
                           / ((from_mat.shape[1] / skew.sum(1, keepdims = True)) * skew)) / from_mat.shape[0],
                      {'from_mat': 'target_outputs',
                       'of_mat': 'predicted_outputs',
                       'skew': 'multi_class_skewnesses'}]}

    return Piece(forwards, backwards)



def PIECE___rbm_energy():

    forwards = 1

    backwards = 1

    return Piece(forwards, backwards)