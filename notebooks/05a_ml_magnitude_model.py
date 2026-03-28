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
from pyspark.sql.functions import col, abs as spark_abs, current_timestamp
from pyspark.sql.types import (StructType, StructField, StringType,
                                FloatType, DoubleType, IntegerType)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ── Session ──────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName('SeismicML_Magnitude') \
    .config('spark.cassandra.connection.host', '127.0.0.1') \
    .config('spark.cassandra.connection.port', '9042') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.jars.packages',
            'com.datastax.spark:spark-cassandra-connector_2.12:3.5.0') \
    .config('spark.driver.extraJavaOptions',
            '-Dio.netty.tryReflectionSetAccessible=true') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'✅ Spark {spark.version} — Magnitude Prediction Model')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5a.1 — Load clean events from disk
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
print(f'Training events loaded: {total:,}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5a.2 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════
feature_cols = ['depth_km', 'gap', 'rms', 'nst', 'sig',
                'latitude', 'longitude', 'tsunami_flag']

feature_df = events_df.select(
    col('event_id'),
    col('mag').alias('label'),
    col('depth_km'),
    col('gap'),
    col('rms'),
    col('nst'),
    col('sig'),
    col('latitude'),
    col('longitude'),
    col('tsunami_flag'),
    col('depth_band'),
).na.fill({
    'gap': 180.0, 'rms': 0.5, 'nst': 5,
    'sig': 0, 'tsunami_flag': 0
}).dropna(subset=['label', 'depth_km', 'latitude', 'longitude'])

print(f'Clean feature rows: {feature_df.count():,}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5a.3 — Build Pipeline
# ═══════════════════════════════════════════════════════════════════════════
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_raw')
scaler    = StandardScaler(inputCol='features_raw', outputCol='features',
                           withMean=True, withStd=True)
rf        = RandomForestRegressor(
    featuresCol='features',
    labelCol='label',
    numTrees=100,
    maxDepth=8,
    seed=42
)

pipeline = Pipeline(stages=[assembler, scaler, rf])

# ═══════════════════════════════════════════════════════════════════════════
# Step 5a.4 — Train / Test Split & Train
# ═══════════════════════════════════════════════════════════════════════════
train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)
print(f'Train: {train_df.count():,}  |  Test: {test_df.count():,}')
print('Training RandomForest... (this takes ~2-3 minutes)')

rf_model = pipeline.fit(train_df)

# ═══════════════════════════════════════════════════════════════════════════
# Step 5a.5 — Evaluate
# ═══════════════════════════════════════════════════════════════════════════
predictions = rf_model.transform(test_df)

rmse = RegressionEvaluator(labelCol='label', predictionCol='prediction',
                            metricName='rmse').evaluate(predictions)
r2   = RegressionEvaluator(labelCol='label', predictionCol='prediction',
                            metricName='r2').evaluate(predictions)
mae  = RegressionEvaluator(labelCol='label', predictionCol='prediction',
                            metricName='mae').evaluate(predictions)

print(f'\n📊 RandomForest Results:')
print(f'   RMSE : {rmse:.4f}  (target < 0.60)')
print(f'   R²   : {r2:.4f}  (target > 0.70)')
print(f'   MAE  : {mae:.4f}')

# Feature importance
rf_best     = rf_model.stages[-1]
importances = list(zip(feature_cols, rf_best.featureImportances))
print('\n📊 Feature Importances:')
for feat, imp in sorted(importances, key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f'   {feat:20s}: {imp:.4f}  {bar}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5a.6 — Save model to disk
# ═══════════════════════════════════════════════════════════════════════════
model_path = 'outputs/models/rf_magnitude'
if os.path.exists(model_path):
    shutil.rmtree(model_path)
rf_model.write().overwrite().save(model_path)
print(f'\n✅ Model saved to: {model_path}')

# ═══════════════════════════════════════════════════════════════════════════
# Step 5a.7 — Write predictions to Cassandra
# ═══════════════════════════════════════════════════════════════════════════
print('\nWriting predictions to Cassandra magnitude_predictions...')

pred_df = predictions.select(
    col('event_id'),
    col('label').alias('actual_mag'),
    col('prediction').alias('predicted_mag'),
    spark_abs(col('label') - col('prediction')).alias('prediction_error'),
    col('depth_km'),
    col('depth_band'),
    col('nst'),
    col('gap'),
    col('rms'),
    current_timestamp().alias('predicted_at')
)

pred_df.write \
    .format('org.apache.spark.sql.cassandra') \
    .option('keyspace', 'seismic') \
    .option('table', 'magnitude_predictions') \
    .mode('append') \
    .save()

pred_count = pred_df.count()
print(f'✅ {pred_count:,} predictions written to Cassandra')

# Sample predictions
print('\nSample Predictions:')
predictions.select(
    'event_id',
    col('label').alias('actual_mag'),
    col('prediction').alias('predicted_mag'),
    col('depth_band')
).show(10)

spark.stop()

print('\n' + '='*55)
print('✅  Week 5a — Magnitude Prediction COMPLETE!')
print(f'   Training events  : {total:,}')
print(f'   RMSE             : {rmse:.4f}')
print(f'   R²               : {r2:.4f}')
print(f'   MAE              : {mae:.4f}')
print(f'   Predictions saved: {pred_count:,} rows → Cassandra')
print(f'   Model saved      : {model_path}')
print('='*55)