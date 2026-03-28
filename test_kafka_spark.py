import os, sys
os.environ['JAVA_HOME']   = 'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.18.8-hotspot'
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('KafkaTest') \
    .config('spark.jars.packages',
            'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')

df = spark.read \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'seismic-events') \
    .option('startingOffsets', 'earliest') \
    .load()

print(f'Row count: {df.count()}')
spark.stop()