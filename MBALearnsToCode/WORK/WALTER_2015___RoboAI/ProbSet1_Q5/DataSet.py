import pandas


class DataSet(object):
    def __init__(self, file_name):
        data_frame = pandas.io.parsers.read_csv(file_name)
        data_frame['xy'] = data_frame.apply(lambda row: (as_integer(row[0]), as_integer(row[1])), axis=1)
        data_frame['ob'] = data_frame.iloc[:, 2]
        data_frame = data_frame[['xy', 'ob']]
        data_sequences = []
        while len(data_frame) > 0:
            i = 0
            num_rows = len(data_frame)
            while (i < num_rows - 1) & isinstance(data_frame.iloc[i]['ob'], str):
                i += 1
            if i == num_rows - 1:
                i += 1
            data_sequences += [data_frame.iloc[:i]]
            data_frame.drop(data_frame.index[:(i + 1)], inplace=True)
        self.data_sequences = data_sequences


def as_integer(obj):
    try:
        obj = int(obj)
    except:
        pass
    return obj