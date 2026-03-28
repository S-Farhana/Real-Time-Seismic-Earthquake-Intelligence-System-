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
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                                FloatType, IntegerType, DoubleType)

# ── Session with Cassandra connector only (no Kafka JAR) ────────────────────
spark = SparkSession.builder \
    .appName('SeismicCassandraWrite') \
    .config('spark.cassandra.connection.host', '127.0.0.1') \
    .config('spark.cassandra.connection.port', '9042') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.jars.packages',
            'com.datastax.spark:spark-cassandra-connector_2.12:3.5.0') \
    .config('spark.driver.extraJavaOptions',
            '-Dio.netty.tryReflectionSetAccessible=true') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'Spark version: {spark.version}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 3.1 — Load clean events from disk (saved by Week 2 script)
# ═══════════════════════════════════════════════════════════════════════════
output_path = 'outputs/clean_events'

schema = StructType([
    StructField('event_id',     StringType(),  True),
    StructField('mag',          FloatType(),   True),
    StructField('depth_km',     DoubleType(),  True),
    StructField('latitude',     DoubleType(),  True),
    StructField('longitude',    DoubleType(),  True),
    StructField('depth_band',   StringType(),  True),
    StructField('mag_class',    StringType(),  True),
    StructField('gap',          FloatType(),   True),
    StructField('rms',          FloatType(),   True),
    StructField('nst',          IntegerType(), True),
    StructField('sig',          IntegerType(), True),
    StructField('tsunami_flag', IntegerType(), True),
    StructField('net',          StringType(),  True),
    StructField('time_iso',     StringType(),  True),
])

clean_df = spark.read.json(output_path, schema=schema)
total = clean_df.count()
print(f'\nLoaded {total:,} clean events from disk')

# ═══════════════════════════════════════════════════════════════════════════
# Step 3.2 — Prepare DataFrame to match Cassandra events table schema
#
# Cassandra events table PRIMARY KEY: ((depth_band, event_date), mag, event_id)
# Required columns: depth_band, event_date, mag, event_id,
#                   depth_km, gap, latitude, longitude, mag_class,
#                   net, nst, rms, sig, time_iso, tsunami_flag
# ═══════════════════════════════════════════════════════════════════════════

cassandra_df = clean_df.withColumn(
    'event_date',
    F.when(
        F.col('time_iso').isNotNull() & (F.col('time_iso') != ''),
        F.to_date(F.col('time_iso'))
    ).otherwise(F.current_date())
).select(
    'depth_band',
    'event_date',
    'mag',
    'event_id',
    'depth_km',
    'gap',
    F.lit('').alias('ingested_at'),
    'latitude',
    'longitude',
    'mag_class',
    'net',
    'nst',
    F.lit('').alias('place'),
    'rms',
    'sig',
    'time_iso',
    'tsunami_flag'
).filter(
    F.col('depth_band').isNotNull() &
    F.col('event_date').isNotNull() &
    F.col('mag').isNotNull() &
    F.col('event_id').isNotNull()
)

write_count = cassandra_df.count()
print(f'Events ready to write: {write_count:,}')

# Preview what we're writing
print('\nSample rows to write:')
cassandra_df.select(
    'event_id', 'mag', 'depth_band', 'event_date', 'mag_class', 'net'
).show(5, truncate=40)

# ═══════════════════════════════════════════════════════════════════════════
# Step 3.3 — Write to Cassandra
# ═══════════════════════════════════════════════════════════════════════════
print('Writing to Cassandra seismic.events ...')

cassandra_df.write \
    .format('org.apache.spark.sql.cassandra') \
    .option('keyspace', 'seismic') \
    .option('table', 'events') \
    .mode('append') \
    .save()

print('✅ Write complete!')

# ═══════════════════════════════════════════════════════════════════════════
# Step 3.4 — Verify read-back from Cassandra
# ═══════════════════════════════════════════════════════════════════════════
print('\nVerifying read-back from Cassandra...')

verify_df = spark.read \
    .format('org.apache.spark.sql.cassandra') \
    .option('keyspace', 'seismic') \
    .option('table', 'events') \
    .load()

verified_count = verify_df.count()
print(f'✅ Rows readable from Cassandra: {verified_count:,}')

print('\nSample from Cassandra:')
verify_df.select(
    'event_id', 'mag', 'depth_band', 'event_date', 'mag_class', 'net'
).show(10, truncate=40)

print('\nDepth band distribution in Cassandra:')
verify_df.groupBy('depth_band') \
    .agg(F.count('*').alias('count')) \
    .orderBy('depth_band') \
    .show()

print('\nTop networks in Cassandra:')
verify_df.groupBy('net') \
    .agg(F.count('*').alias('count')) \
    .orderBy(F.desc('count')) \
    .show(10)

spark.stop()

print('\n' + '='*55)
print('✅  Week 3 NoSQL Storage COMPLETE!')
print(f'   Events written to Cassandra : {write_count:,}')
print(f'   Events verified (read-back) : {verified_count:,}')
print('   Table : seismic.events')
print('='*55)