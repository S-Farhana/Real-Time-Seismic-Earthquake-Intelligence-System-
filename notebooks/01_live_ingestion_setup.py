import os
import sys
import subprocess

# Force Java 17
os.environ['JAVA_HOME']   = 'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.18.8-hotspot'
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH']        = (
    'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.18.8-hotspot\\bin;'
    'C:\\hadoop\\bin;'
    + os.environ.get('PATH', '')
)

sys.path.append('.')
from config.spark_config import get_spark_session
from pyspark.sql.functions import col, get_json_object

spark = get_spark_session('SeismicVerify')
spark.sparkContext.setLogLevel('WARN')
print(f'✅ Spark version: {spark.version}')

# Read directly from local Kafka
df = spark.read \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'seismic-events') \
    .option('startingOffsets', 'earliest') \
    .load()

print(f'✅ Messages in topic: {df.count()}')

df_parsed = df.select(
    col('key').cast('string').alias('event_id'),
    col('value').cast('string').alias('raw_json'),
    col('timestamp')
)

df_preview = df_parsed.select(
    'event_id',
    'timestamp',
    get_json_object('raw_json', '$.mag').cast('float').alias('magnitude'),
    get_json_object('raw_json', '$.place').alias('place'),
    get_json_object('raw_json', '$.depth_km').cast('float').alias('depth_km')
)

df_preview.show(10, truncate=50)
print('✅ Week 1 Step 1.4 — COMPLETE')

spark.stop()