import os
import sys

# Set BEFORE PySpark imports — critical
os.environ['JAVA_HOME']             = 'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.18.8-hotspot'
os.environ['HADOOP_HOME']           = 'C:\\hadoop'
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PATH'] = (
    'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.18.8-hotspot\\bin;'
    'C:\\hadoop\\bin;'
    + os.environ.get('PATH', '')
)

from pyspark.sql import SparkSession

def get_spark_session(app_name='SeismicIntelligence'):
    return SparkSession.builder \
        .appName(app_name) \
        .config('spark.cassandra.connection.host', '127.0.0.1') \
        .config('spark.sql.shuffle.partitions', '4') \
        .config('spark.jars.packages',
                'com.datastax.spark:spark-cassandra-connector_2.12:3.5.0,'
                'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
        .config('spark.driver.extraJavaOptions',
                '-Dio.netty.tryReflectionSetAccessible=true '
                '-Djdk.attach.allowAttachSelf=true') \
        .config('spark.executor.extraJavaOptions',
                '-Dio.netty.tryReflectionSetAccessible=true') \
        .config('spark.python.worker.reuse', 'false') \
        .config('spark.executor.heartbeatInterval', '60s') \
        .config('spark.network.timeout', '300s') \
        .getOrCreate()