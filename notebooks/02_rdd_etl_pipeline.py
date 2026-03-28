import os
import sys
import shutil

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
                                FloatType, IntegerType, LongType)

# ── Session (Kafka only — no Cassandra JAR) ─────────────────────────────────
spark = SparkSession.builder \
    .appName('SeismicETL') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.jars.packages',
            'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
    .config('spark.driver.extraJavaOptions',
            '-Dio.netty.tryReflectionSetAccessible=true') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'Spark version: {spark.version}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 2.1 — Load Kafka Events
# ═══════════════════════════════════════════════════════════════════════════
raw_df = spark.read \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'seismic-events') \
    .option('startingOffsets', 'earliest') \
    .load()

# ACTION 1 — total raw messages
total_raw = raw_df.count()
print(f'\nACTION 1 — Total raw messages: {total_raw}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 2.2 — Transformation Chain (9 transformations, 8 actions)
# ═══════════════════════════════════════════════════════════════════════════

# TRANSFORMATION 1: decode bytes → JSON string
json_df = raw_df.select(F.col('value').cast('string').alias('raw_json'))

# TRANSFORMATION 2: parse JSON into columns
schema = StructType([
    StructField('event_id',    StringType(), True),
    StructField('mag',         FloatType(),  True),
    StructField('place',       StringType(), True),
    StructField('time_iso',    StringType(), True),
    StructField('longitude',   FloatType(),  True),
    StructField('latitude',    FloatType(),  True),
    StructField('depth_km',    FloatType(),  True),
    StructField('gap',         FloatType(),  True),
    StructField('rms',         FloatType(),  True),
    StructField('nst',         IntegerType(),True),
    StructField('sig',         IntegerType(),True),
    StructField('net',         StringType(), True),
    StructField('event_type',  StringType(), True),
    StructField('tsunami',     IntegerType(),True),
    StructField('mag_type',    StringType(), True),
])

parsed_df = json_df.select(
    F.from_json(F.col('raw_json'), schema).alias('d')
).select('d.*')

# TRANSFORMATION 3: filter only earthquakes
quake_df = parsed_df.filter(F.col('event_type') == 'earthquake')

# TRANSFORMATION 4: drop nulls on critical fields
valid_df = quake_df.filter(
    F.col('mag').isNotNull() &
    F.col('latitude').isNotNull() &
    F.col('longitude').isNotNull() &
    F.col('depth_km').isNotNull()
)

# ACTION 2 — valid earthquake count
total_valid = valid_df.count()
print(f'ACTION 2 — Valid earthquake events: {total_valid}')

# TRANSFORMATION 5: classify depth band
depth_df = valid_df.withColumn('depth_band',
    F.when(F.col('depth_km') < 70,  'shallow')
     .when(F.col('depth_km') < 300, 'intermediate')
     .otherwise('deep')
)

# TRANSFORMATION 6: classify magnitude class
classified_df = depth_df.withColumn('mag_class',
    F.when(F.col('mag') < 2.5, 'micro')
     .when(F.col('mag') < 4.0, 'minor')
     .when(F.col('mag') < 5.0, 'light')
     .when(F.col('mag') < 6.0, 'moderate')
     .when(F.col('mag') < 7.0, 'strong')
     .otherwise('major')
).withColumn('tsunami_flag', F.coalesce(F.col('tsunami'), F.lit(0))) \
 .withColumn('gap',          F.coalesce(F.col('gap'),     F.lit(180.0))) \
 .withColumn('rms',          F.coalesce(F.col('rms'),     F.lit(0.0))) \
 .withColumn('nst',          F.coalesce(F.col('nst'),     F.lit(0))) \
 .withColumn('net',          F.coalesce(F.col('net'),     F.lit('unknown')))

# ACTION 3 — sample events
print('\nACTION 3 — Sample Events:')
classified_df.select(
    'event_id', 'mag', 'depth_km', 'mag_class', 'depth_band', 'net'
).show(3, truncate=40)

# TRANSFORMATION 7: network event counts
network_counts = classified_df.groupBy('net') \
    .agg(F.count('*').alias('event_count')) \
    .orderBy(F.desc('event_count'))

# ACTION 4 — top 10 networks
print('ACTION 4 — Top Seismic Networks:')
network_counts.show(10, truncate=20)

# TRANSFORMATION 8: depth band summary
depth_summary = classified_df.groupBy('depth_band') \
    .agg(F.count('*').alias('event_count'))

# ACTION 5 — depth band counts
print('ACTION 5 — Depth Band Summary:')
depth_summary.show()

# TRANSFORMATION 9: avg magnitude per class
mag_stats = classified_df.groupBy('mag_class') \
    .agg(
        F.round(F.avg('mag'), 3).alias('avg_mag'),
        F.count('*').alias('count')
    ).orderBy('mag_class')

# ACTION 6 — magnitude class averages
print('ACTION 6 — Avg Magnitude Per Class:')
mag_stats.show()

# ACTION 7 — join classified events with depth band counts
enriched_df = classified_df.join(
    depth_summary.withColumnRenamed('event_count', 'band_total'),
    on='depth_band',
    how='left'
)
print(f'ACTION 7 — Joined DataFrame count: {enriched_df.count()}')

# ACTION 8 — magnitude range
mag_agg = classified_df.agg(
    F.round(F.min('mag'), 2).alias('min_mag'),
    F.round(F.max('mag'), 2).alias('max_mag'),
    F.round(F.avg('mag'), 2).alias('mean_mag')
).collect()[0]

print(f'ACTION 8 — Mag range: {mag_agg.min_mag} – {mag_agg.max_mag}, '
      f'Mean: {mag_agg.mean_mag}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 2.3 — Save clean events to disk
# ═══════════════════════════════════════════════════════════════════════════
output_path = 'outputs/clean_events'
if os.path.exists(output_path):
    shutil.rmtree(output_path)
    print(f'\nCleared old output: {output_path}')

# Select final clean columns to save
clean_df = classified_df.select(
    'event_id', 'mag', 'depth_km', 'latitude', 'longitude',
    'depth_band', 'mag_class', 'gap', 'rms', 'nst', 'sig',
    'tsunami_flag', 'net', 'time_iso'
)

clean_df.write.mode('overwrite').json(output_path)
print(f'Clean events saved to: {output_path}')

spark.stop()

print('\n' + '='*55)
print('✅  Week 2 ETL Pipeline COMPLETE!')
print(f'   Raw messages      : {total_raw}')
print(f'   Valid earthquakes : {total_valid}')
print(f'   Saved to          : {output_path}')
print('   9 Transformations + 8 Actions demonstrated!')
print('='*55)