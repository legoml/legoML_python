# SETUP
from MBALearnsToCode.Spark.SPARK___init import import_pyspark
pyspark = import_pyspark('/Applications/spark-1.4.0')
sc = pyspark.SparkContext()
from pyspark import SQLContext
sqlContext = SQLContext(sc)
from pyspark.mllib.regression import LabeledPoint

# OTHER IMPORTS
from collections import OrderedDict


# PARALLELIZE
data = range(20)
rdd = sc.parallelize(data)


# TAKE & SAMPLE
rdd.take(5)
rdd.takeSample(withReplacement=True, num=5, seed=123)


# BASIC MAP-REDUCE
dataSq___map = rdd.map(lambda x: x ** 2)
dataSum = rdd.reduce(lambda a, b: a + b)


# RDD from 0 to 100
data = range(100)
rdd = sc.parallelize(data)
transformed_rdd = rdd.map(lambda x: (3 * x if x % 2 else 2 * x))
transformed_rdd.take(10)


# FILTER by LOGICAL CONDITION
rdd.filter(lambda x: not(x % 2)).take(10)


# IRIS DATA
iris_data_file_path = '/Applications/Dropbox/iris.data'
def parse_text_file_line(text_file_line):
    from collections import OrderedDict
    from numpy import array
    variables = text_file_line.split(',')
    x = array(variables[:4], dtype=float)
    y = variables[4]
    return OrderedDict(x=x, y=y)
iris_data = sc.textFile(iris_data_file_path).map(parse_text_file_line)
test_data_case = iris_data.take(1)[0]

X = iris_data.map(lambda ordered_dict: ordered_dict['x'])
num_cases = X.count()
#demeaned_X = X.