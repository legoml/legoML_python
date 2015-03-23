from numpy import array, atleast_2d, exp, log, sqrt
from MachinePlayground.Classes import Piece



def PIECE___average_half_square_error():
    """PIECE_average_half_squared_error

    AVERAGE_HALF_SQUARE_ERROR = 1/2 x sum of squared differences between items in PREDICTED_OUTPUTS multi-dimensional
    array and TARGET_OUTPUTS multi-dimensional array

    ROOT_MEAN_SQUARE_ERROR = square root of 2 x AVERAGE_HALF_SQUARED_ERROR
    """

    forwards = {'average_half_square_error':
                    [lambda a0, a1: ((a1 - a0) ** 2).sum() / (2 * a0.shape[0]),
                     {'a0': 'target_outputs',
                      'a1': 'predicted_outputs'}],

                'root_mean_square_error':
                    [lambda a0, a1: sqrt(((a1 - a0) ** 2).sum() / a0.shape[0]),
                     {'a0': 'target_outputs',
                      'a1': 'predicted_outputs'}]}

    backwards = {('DOVERD', 'average_half_square_error', 'predicted_outputs'):
                     [lambda a0, a1: (a1 - a0) / a0.shape[0],
                      {'a0': 'target_outputs',
                       'a1': 'predicted_outputs'}],

                 ('DOVERD', 'root_mean_square_error', 'predicted_outputs'):
                     [lambda a0, a1: ((a1 - a0) / a0.shape[0]) / rmse,
                      {'a0': 'target_outputs',
                       'a1': 'predicted_outputs',
                       'rmse': 'root_mean_square_error'}]}

    return Piece(forwards, backwards)



def PIECE___average_unskewed_cross_entropy_binary_classes():

    forwards = {'average_unskewed_cross_entropy_binary_classes':
                    [lambda from_arr, of_arr, pos_skew = array([1]), tiny = exp(-36):
                        - ((from_arr * log(of_arr + tiny)) / pos_skew
                        + ((1. - from_arr) * log(1. - of_arr + tiny)) / (2. - pos_skew)).sum() / from_arr.shape[0],
                    {'from_arr': 'target_outputs',
                     'of_arr': 'predicted_outputs',
                     'pos_skew': 'positive_class_skewnesses'}]}

    backwards = {('DOVERD', 'average_unskewed_cross_entropy_binary_classes', 'predicted_outputs'):
                    [lambda from_arr, of_arr, pos_skew = array([1]):
                        - ((from_arr / of_arr) / pos_skew
                        - ((1. - from_arr) / (1. - of_arr)) / (2. - pos_skew)) / from_arr.shape[0],
                     {'from_arr': 'target_outputs',
                      'of_arr': 'predicted_outputs',
                      'pos_skew': 'positive_class_skewnesses'}]}

    return Piece(forwards, backwards)



def PIECE___average_unskewed_cross_entropy_multi_classes():

    forwards = {'average_unskewed_cross_entropy_multi_classes':
                    [lambda from_mat, of_mat, skew = atleast_2d(array([1])), tiny = exp(-36):
                        - ((from_mat * log(of_mat + tiny))
                           / ((from_mat.shape[1] / skew.sum(1, keepdims = True)) * skew)).sum() / from_mat.shape[0],
                     {'from_mat': 'target_outputs',
                      'of_mat': 'predicted_outputs',
                      'skew': 'class_skewnesses'}]}

    backwards = {('DOVERD', 'average_unskewed_cross_entropy_multi_classes', 'predicted_outputs'):
                     [lambda from_mat, of_mat, skew = atleast_2d(array([1])):
                        - ((from_mat / of_mat)
                           / ((from_mat.shape[1] / skew.sum(1, keepdims = True)) * skew)) / from_mat.shape[0],
                      {'from_mat': 'target_outputs',
                       'ofMat': 'hypoOutputs___matrixRowsForCases',
                       'skew': 'class_skewnesses'}]}

    return Piece(forwards, backwards)