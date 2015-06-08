from MBALearnsToCode.Spark.SPARK___init import import_pyspark

pyspark = import_pyspark()
sc = pyspark.SparkContext()
sql_context = pyspark.sql.SQLContext(sc)

data_folder =
