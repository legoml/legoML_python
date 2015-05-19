from os import listdir, path
from pandas.io.parsers import read_csv
from numpy import abs, array, arctan2, cos, isnan, minimum, nan, pi, sign, sin, sqrt


class DriverTripData:
    def __init__(self, data_folder_path):
        d = {}
        for driver_id in listdir(data_folder_path):
            driver = int(driver_id)
            d[driver] = set()
            trips_folder_path = data_folder_path + '/' + driver_id
            for trip_csv_file_name in listdir(trips_folder_path):
                d[driver].add(int(path.splitext(trip_csv_file_name)[0]))
        self.data_folder_path = data_folder_path
        self.dict = d

    def load(self, driver, trip):
        return read_csv(self.data_folder_path + '/' + str(driver) + '/' + str(trip) + '.csv')


def calc_velocity_vector(trip_data_frame):
    trip_data_frame['dx'] = trip_data_frame.x.diff(1)
    trip_data_frame['dy'] = trip_data_frame.y.diff(1)
    trip_data_frame.ix[0, ['dx', 'dy']] = trip_data_frame.ix[1, ['dx', 'dy']]


def calc_acceleration_vector(trip_data_frame):
    trip_data_frame['ddx'] = trip_data_frame.dx.diff(1)
    trip_data_frame['ddy'] = trip_data_frame.dy.diff(1)
    trip_data_frame.ix[0, ['ddx', 'ddy']] = trip_data_frame.ix[1, ['ddx', 'ddy']]


def calc_velocity(trip_data_frame):
    trip_data_frame['velocity'] = sqrt(trip_data_frame.dx ** 2 + trip_data_frame.dy ** 2)


def calc_acceleration(trip_data_frame):
    trip_data_frame['acceleration'] = sqrt(trip_data_frame.ddx ** 2 + trip_data_frame.ddy ** 2)


def calc_angle(trip_data_frame):
    trip_data_frame['angle'] = arctan2(trip_data_frame.dy, trip_data_frame.dx)


def calc_angular_velocity(trip_data_frame):
    angular_velocity = trip_data_frame.angle.diff(1)
    trip_data_frame['angular_velocity'] = arctan2(sin(angular_velocity), cos(angular_velocity))
    trip_data_frame.ix[0, 'angular_velocity'] = trip_data_frame.ix[1, 'angular_velocity']


def calc_angular_acceleration(trip_data_frame):
    trip_data_frame['angular_acceleration'] = trip_data_frame.angular_velocity.diff(1)
    trip_data_frame.ix[0, 'angular_acceleration'] = trip_data_frame.ix[1, 'angular_acceleration']


def calc_velocity_validity(trip_data_frame, max_velocity_m_per_s=50):
    trip_data_frame['check_velocity'] = trip_data_frame.velocity < max_velocity_m_per_s


def calc_angular_velocity_validity(trip_data_frame, max_absolute_angular_velocity=.75 * pi):
    trip_data_frame['check_angular_velocity'] = abs(trip_data_frame.angular_velocity) < max_absolute_angular_velocity


def calc_trip_data(trip_data_frame, max_velocity_m_per_s=50, max_absolute_angular_velocity=.75 * pi):
    calc_velocity_vector(trip_data_frame)
    calc_acceleration_vector(trip_data_frame)
    calc_velocity(trip_data_frame)
    calc_acceleration(trip_data_frame)
    calc_angle(trip_data_frame)
    calc_angular_velocity(trip_data_frame)
    calc_angular_acceleration(trip_data_frame)
    calc_velocity_validity(trip_data_frame, max_velocity_m_per_s)
    calc_angular_velocity_validity(trip_data_frame, max_absolute_angular_velocity)


def check_trip_data_quality(trip_data_frame):
    return trip_data_frame.check_velocity.all() & trip_data_frame.check_angular_velocity.all()


def clean_trip_data_frame(trip_data_frame):
    return trip_data_frame
