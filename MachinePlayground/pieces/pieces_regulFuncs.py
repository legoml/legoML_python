from numpy import sign
from MachinePlayground.Classes import Piece


def piece_regulL1():

    forwards = {'weightPenalty':
                    [lambda w: abs(w).sum(),
                     {'w': 'weights'}]}

    backwards = {('dOverD', 'weightPenalty', 'weights'):
                     [lambda w: sign(w),
                      {'w': 'weights'}]}

    return Piece(forwards, backwards)


def piece_regulL2():

    forwards = {'weightPenalty':
                    [lambda w: (w ** 2).sum() / 2,
                     {'w': 'weights'}]}

    backwards = {('dOverD', 'weightPenalty', 'weights'):
                     [lambda w: w,
                      {'w': 'weights'}]}

    return Piece(forwards, backwards)