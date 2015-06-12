from os import environ
from sys import exit, path


def import_pyspark(spark_home='C:/Programs/spark-1.3.1'):
    environ['SPARK_HOME'] = spark_home
    path.append(spark_home + '/python')
    try:
        import pyspark
        return pyspark
    except ImportError as e:
        print ("Cannot Import PySpark:", e)
        exit(1)
