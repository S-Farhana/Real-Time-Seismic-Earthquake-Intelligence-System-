from pyspark.sql import SparkSession

def get_spark_session(app_name='SeismicIntelligence'):
    return SparkSession.builder \
        .appName(app_name) \
        .config('spark.cassandra.connection.host', '127.0.0.1') \
        .config('spark.sql.shuffle.partitions', '4') \
        .config('spark.jars.packages',
                'com.datastax.spark:spark-cassandra-connector_2.12:3.5.0,'
                'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
        .getOrCreate()