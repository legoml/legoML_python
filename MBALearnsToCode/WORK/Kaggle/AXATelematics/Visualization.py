from ggplot import *
from numpy import nan
from MBALearnsToCode.WORK.Kaggle.AXATelematics.Data_and_Features import check_trip_data_quality


def plot_trip(trip_data_frame, details=tuple(), **kwargs):
    if check_trip_data_quality(trip_data_frame):
        plot = ggplot(trip_data_frame, aes(x='x', y='y')) + geom_point(**kwargs)
    else:
        d = trip_data_frame.copy()
        T = len(d)
        d['bad_x'] = T * [0]
        d['bad_y'] = T * [0]
        bad = ~d.check_velocity | ~d.check_angular_velocity
        d.ix[bad, ['bad_x', 'bad_y']] = d.ix[bad, ['x', 'y']]
        plot = ggplot(d, aes('x', 'y')) + geom_point(**kwargs) +\
            geom_point(aes('bad_x', 'bad_y'), color='red', size=90)
    return plot