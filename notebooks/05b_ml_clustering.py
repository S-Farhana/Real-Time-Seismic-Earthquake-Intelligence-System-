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
from pyspark.sql.functions import (col, avg, max, count, stddev,
                                    round as spark_round)
from pyspark.sql.types import (StructType, StructField, StringType,
                                FloatType, DoubleType, IntegerType)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ── Session ──────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName('SeismicML_Clustering') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.jars.packages',
            'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
    .config('spark.driver.extraJavaOptions',
            '-Dio.netty.tryReflectionSetAccessible=true') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'✅ Spark {spark.version} — KMeans Seismic Clustering')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.1 — Load clean events from disk
# ═══════════════════════════════════════════════════════════════════════════
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

events_df = spark.read.json('outputs/clean_events', schema=schema)
total = events_df.count()
print(f'Events loaded: {total:,}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.2 — Feature Engineering for Clustering
# ═══════════════════════════════════════════════════════════════════════════
cluster_df = events_df.select(
    col('event_id'),
    col('mag'),
    col('depth_km'),
    col('latitude'),
    col('longitude'),
    col('depth_band'),
    col('mag_class'),
    col('net'),
).na.fill({
    'mag': 0.0, 'depth_km': 0.0
}).dropna(subset=['mag', 'depth_km', 'latitude', 'longitude'])

print(f'Clean cluster rows: {cluster_df.count():,}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.3 — Build KMeans Pipeline
# ═══════════════════════════════════════════════════════════════════════════
cluster_cols = ['latitude', 'longitude', 'depth_km', 'mag']

cluster_assembler = VectorAssembler(
    inputCols=cluster_cols,
    outputCol='cluster_features'
)

kmeans = KMeans(
    featuresCol='cluster_features',
    predictionCol='cluster_id',
    k=8,
    maxIter=20,
    seed=42
)

cluster_pipeline = Pipeline(stages=[cluster_assembler, kmeans])

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.4 — Train KMeans
# ═══════════════════════════════════════════════════════════════════════════
print('Training KMeans (8 clusters = major tectonic zones)...')
cluster_model = cluster_pipeline.fit(cluster_df)
clustered_df  = cluster_model.transform(cluster_df)

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.5 — Evaluate
# ═══════════════════════════════════════════════════════════════════════════
sil_eval   = ClusteringEvaluator(
    featuresCol='cluster_features',
    predictionCol='cluster_id'
)
silhouette = sil_eval.evaluate(clustered_df)
print(f'\n📊 KMeans Results:')
print(f'   Silhouette score: {silhouette:.4f}  (target > 0.40)')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.6 — Cluster Statistics
# ═══════════════════════════════════════════════════════════════════════════
print('\n📊 Cluster Statistics (ordered by avg magnitude):')
cluster_stats = clustered_df.groupBy('cluster_id').agg(
    count('*').alias('n_events'),
    spark_round(avg('mag'), 3).alias('avg_mag'),
    spark_round(max('mag'), 3).alias('max_mag'),
    spark_round(avg('depth_km'), 1).alias('avg_depth_km'),
    spark_round(avg('latitude'), 2).alias('centroid_lat'),
    spark_round(avg('longitude'), 2).alias('centroid_lon')
).orderBy('avg_mag', ascending=False)
cluster_stats.show(8)

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.7 — Anomaly Detection (mag > cluster mean + 2 std)
# ═══════════════════════════════════════════════════════════════════════════
print('📊 Anomaly Detection (mag > cluster mean + 2σ):')

stats = clustered_df.groupBy('cluster_id').agg(
    avg('mag').alias('mean_mag'),
    stddev('mag').alias('std_mag')
)

anomalies = clustered_df.join(stats, 'cluster_id') \
    .filter(col('mag') > col('mean_mag') + 2 * col('std_mag'))

anomaly_count = anomalies.count()
print(f'   Anomalous events detected: {anomaly_count:,}')

anomalies.select(
    'event_id',
    col('mag').alias('magnitude'),
    'cluster_id',
    spark_round(col('mean_mag'), 3).alias('cluster_mean'),
    'latitude',
    'longitude',
    'depth_band'
).orderBy('magnitude', ascending=False).show(10, truncate=40)

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b.8 — Cluster distribution by depth band
# ═══════════════════════════════════════════════════════════════════════════
print('📊 Cluster Distribution by Depth Band:')
clustered_df.groupBy('cluster_id', 'depth_band') \
    .agg(count('*').alias('count')) \
    .orderBy('cluster_id', 'depth_band') \
    .show(24)

# Save clustered data
output_path = 'outputs/clustered_events'
import shutil
if os.path.exists(output_path):
    shutil.rmtree(output_path)
clustered_df.select(
    'event_id', 'mag', 'depth_km', 'latitude', 'longitude',
    'depth_band', 'mag_class', 'net', 'cluster_id'
).write.mode('overwrite').json(output_path)
print(f'\n✅ Clustered events saved to: {output_path}')

spark.stop()

print('\n' + '='*55)
print('✅  Week 5b — KMeans Clustering COMPLETE!')
print(f'   Events clustered : {total:,}')
print(f'   Clusters         : 8 (tectonic zones)')
print(f'   Silhouette score : {silhouette:.4f}')
print(f'   Anomalies found  : {anomaly_count:,}')
print('='*55)