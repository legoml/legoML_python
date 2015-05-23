from MBALearnsToCode.Spark.SPARK___init import import_pyspark
from MBALearnsToCode.WORK.Kaggle.AXATelematics.Data_and_Features import DriverTripData, calc_trip_data,\
    check_trip_data_quality
from MBALearnsToCode.WORK.Kaggle.AXATelematics.Visualization import plot_trip

pyspark = import_pyspark()
sc = pyspark.SparkContext()
sql_context = pyspark.sql.SQLContext(sc)

data_folder_path = 'C:/Cloud/Dropbox/MBALearnsToCode/data/kaggle_axa_driver_telematics_analysis/drivers'
driver_trip_data = DriverTripData(data_folder_path)
d = driver_trip_data.load(1, 1)
calc_trip_data(d)
check_trip_data_quality(d)
plot_trip(d, ('velocity',), size=1) #