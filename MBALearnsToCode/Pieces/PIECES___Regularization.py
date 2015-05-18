from numpy import sign

from MBALearnsToCode import Piece


def PIECE___l1_weight_regularization():
    """PIECE___l1_weight_regularization:

    WEIGHT_PENALTY = sum of absolute values of items of WEIGHTS multi-dimensional array
    """

    forwards = {'weight_penalty':
                    [lambda w: abs(w).sum(),
                     {'w': 'weights'}]}

    backwards = {('DOVERD', 'weight_penalty', 'weights'):
                     [lambda w: sign(w),
                      {'w': 'weights'}]}

    return Piece(forwards, backwards)



def PIECE___l2_weight_regularization():
    """PIECE___l2_weight_regularization:

    WEIGHT_PENALTY = 1/2 x sum of squares of items of WEIGHTS multi-dimensional array
    """

    forwards = {'weight_penalty':
                    [lambda w: (w ** 2).sum() / 2,
                     {'w': 'weights'}]}

    backwards = {('DOVERD', 'weight_penalty', 'weights'):
                     [lambda w: w,
                      {'w': 'weights'}]}

    return Piece(forwards, backwards)