from numpy import array, atleast_2d, atleast_3d, concatenate, delete, diag, exp, ndarray, ones, squeeze, tanh, zeros
from MachinePlayground.Classes import Piece

def PIECE___read_csv():
    import csv
    forwards = {'data_frame':
                    [lambda f, dialect, fmt_params:
                        csv.reader(f, dialect = dialect, **fmt_params),
                     {'f': 'csv_file',
                      'dialect': 'dialect',
                      'fmt_params': 'formatting_parameters'}]}
    return Piece(forwards)


