import os
import sys


def import_pyspark(spark_home='C:/Programs/spark-1.3.1'):
    os.environ['SPARK_HOME'] = spark_home
    sys.path.append(spark_home + '/python')
    try:
        import pyspark
        return pyspark
    except ImportError as e:
        print ("Cannot Import PySpark:", e)
        sys.exit(1)
