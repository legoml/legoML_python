import os
import sys


def spark_context(spark_home):
    os.environ['SPARK_HOME'] = spark_home
    sys.path.append(spark_home + '/python')
    try:
        from pyspark import SparkContext
        from pyspark import SparkConf
        print ("Successfully imported Spark Modules")
        return SparkContext()
    except ImportError as e:
        print ("Can not import Spark Modules", e)
        sys.exit(1)