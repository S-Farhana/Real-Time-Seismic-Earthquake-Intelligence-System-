import os
import sys

# ── Environment setup ───────────────────────────────────────────────────────
os.environ['JAVA_HOME']             = 'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.18.8-hotspot'
os.environ['HADOOP_HOME']           = 'C:\\hadoop'
os.environ['PATH']                  = (
    'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.18.8-hotspot\\bin;'
    'C:\\hadoop\\bin;'
    + os.environ.get('PATH', '')
)
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, to_date,
    when, window, count, max, avg, sum,
    current_timestamp, round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    DoubleType, IntegerType, LongType
)

# ── Session ─────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName('SeismicStreaming') \
    .config('spark.cassandra.connection.host', '127.0.0.1') \
    .config('spark.cassandra.connection.port', '9042') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.jars.packages',
            'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,'
            'com.datastax.spark:spark-cassandra-connector_2.12:3.5.0') \
    .config('spark.driver.extraJavaOptions',
            '-Dio.netty.tryReflectionSetAccessible=true') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'✅ Spark {spark.version} — Structured Streaming started')

# ═══════════════════════════════════════════════════════════════════════════
# Step 4.1 — Schema
# ═══════════════════════════════════════════════════════════════════════════
seismic_schema = StructType([
    StructField('event_id',    StringType(),  True),
    StructField('mag',         FloatType(),   True),
    StructField('place',       StringType(),  True),
    StructField('time_ms',     LongType(),    True),
    StructField('time_iso',    StringType(),  True),
    StructField('latitude',    DoubleType(),  True),
    StructField('longitude',   DoubleType(),  True),
    StructField('depth_km',    DoubleType(),  True),
    StructField('gap',         FloatType(),   True),
    StructField('rms',         FloatType(),   True),
    StructField('nst',         IntegerType(), True),
    StructField('sig',         IntegerType(), True),
    StructField('net',         StringType(),  True),
    StructField('mag_type',    StringType(),  True),
    StructField('event_type',  StringType(),  True),
    StructField('alert',       StringType(),  True),
    StructField('tsunami',     IntegerType(), True),
    StructField('ingested_at', StringType(),  True),
])

# ═══════════════════════════════════════════════════════════════════════════
# Step 4.2 — Read Live Stream from Kafka
# ═══════════════════════════════════════════════════════════════════════════
raw_stream = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'seismic-events') \
    .option('startingOffsets', 'latest') \
    .option('failOnDataLoss', 'false') \
    .load()

events_stream = raw_stream \
    .select(from_json(col('value').cast('string'), seismic_schema).alias('e')) \
    .select('e.*') \
    .filter(col('event_type') == 'earthquake') \
    .filter(col('mag').isNotNull()) \
    .withColumn('event_ts', to_timestamp(col('time_iso'))) \
    .withColumn('depth_band',
        when(col('depth_km') < 70,  'shallow')
        .when(col('depth_km') < 300,'intermediate')
        .otherwise('deep')
    ) \
    .withColumn('tsunami_flag', col('tsunami'))

print('✅ Kafka stream connected')

# ═══════════════════════════════════════════════════════════════════════════
# Step 4.3 — Alert Stream (M4.0+ events → Cassandra alerts table)
# ═══════════════════════════════════════════════════════════════════════════
alert_stream = events_stream \
    .filter(col('mag') >= 4.0) \
    .withColumn('alert_level',
        when(col('mag') >= 7.0, 'red')
        .when(col('mag') >= 6.0, 'orange')
        .when(col('mag') >= 5.0, 'yellow')
        .otherwise('green')
    ) \
    .withColumn('alert_date', to_date(col('event_ts'))) \
    .withColumn('alert_time', col('event_ts')) \
    .select(
        'alert_date', 'alert_time', 'event_id',
        'mag', 'alert_level', 'tsunami_flag',
        'place', 'latitude', 'longitude'
    )

# ═══════════════════════════════════════════════════════════════════════════
# Step 4.4 — Windowed Aggregations → region_stats
# FIX: Use append mode with watermark — window emits only after watermark
#      passes, then written once as append. Cassandra supports append only.
# ═══════════════════════════════════════════════════════════════════════════
windowed_stats = events_stream \
    .withWatermark('event_ts', '15 minutes') \
    .groupBy(
        window('event_ts', '10 minutes', '5 minutes'),
        col('depth_band')
    ) \
    .agg(
        count('*').alias('event_count'),
        max('mag').alias('max_mag'),
        spark_round(avg('mag'), 2).alias('avg_mag'),
        spark_round(avg('depth_km'), 2).alias('avg_depth'),
        sum('tsunami_flag').alias('tsunami_count')
    ) \
    .select(
        col('depth_band').alias('region_code'),
        to_date(col('window.start')).alias('stat_date'),
        col('event_count'),
        col('max_mag'),
        col('avg_mag'),
        col('avg_depth'),
        col('tsunami_count'),
        current_timestamp().alias('updated_at')
    )

# ═══════════════════════════════════════════════════════════════════════════
# Step 4.5 — Write Streams
# ═══════════════════════════════════════════════════════════════════════════

# QUERY 1: Alerts → Cassandra
print('Starting alerts stream → Cassandra...')
query_alerts = alert_stream.writeStream \
    .format('org.apache.spark.sql.cassandra') \
    .option('keyspace', 'seismic') \
    .option('table', 'alerts') \
    .option('checkpointLocation', 'checkpoints/alerts') \
    .outputMode('append') \
    .trigger(processingTime='30 seconds') \
    .start()
print('✅ Alerts stream started')

# QUERY 2: Windowed stats → Cassandra
# FIX: append mode (not update) — Cassandra connector requires this
print('Starting windowed stats stream → Cassandra...')
query_stats = windowed_stats.writeStream \
    .format('org.apache.spark.sql.cassandra') \
    .option('keyspace', 'seismic') \
    .option('table', 'region_stats') \
    .option('checkpointLocation', 'checkpoints/region_stats') \
    .outputMode('append') \
    .trigger(processingTime='30 seconds') \
    .start()
print('✅ Region stats stream started')

# QUERY 3: Console live monitor
print('Starting console stream (live monitor)...')
query_console = events_stream \
    .select('event_id', 'mag', 'place', 'depth_band',
            'depth_km', 'net', 'tsunami_flag', 'event_ts') \
    .writeStream \
    .format('console') \
    .option('truncate', False) \
    .option('numRows', 10) \
    .outputMode('append') \
    .trigger(processingTime='30 seconds') \
    .start()
print('✅ Console stream started')

print()
print('═' * 60)
print('🌍 LIVE SEISMIC STREAMING ACTIVE')
print('   Kafka → Spark → Cassandra (every 30 seconds)')
print('   Alerts    : M4.0+ events → seismic.alerts')
print('   Stats     : 10-min windows → seismic.region_stats')
print('   Console   : all events printed every 30 seconds')
print('   Press Ctrl+C to stop')
print('═' * 60)

spark.streams.awaitAnyTermination()