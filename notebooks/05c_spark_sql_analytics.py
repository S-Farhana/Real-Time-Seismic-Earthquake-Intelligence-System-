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
from pyspark.sql.types import (StructType, StructField, StringType,
                                FloatType, DoubleType, IntegerType)

# ── Session ──────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName('SeismicSQL_Analytics') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.driver.extraJavaOptions',
            '-Dio.netty.tryReflectionSetAccessible=true') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'✅ Spark {spark.version} — Spark SQL Analytics')

# ═══════════════════════════════════════════════════════════════════════════
# Load clean events
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

# Register SQL view
events_df.createOrReplaceTempView('seismic_events')
print('✅ SQL view registered: seismic_events\n')

# ═══════════════════════════════════════════════════════════════════════════
# QUERY 1 — Top 10 seismic hotspot networks by event count
# ═══════════════════════════════════════════════════════════════════════════
print('='*60)
print('QUERY 1 — Top Seismic Networks by Activity')
print('='*60)
spark.sql('''
    SELECT net,
           depth_band,
           COUNT(*)                   AS total_events,
           ROUND(AVG(mag), 3)         AS avg_magnitude,
           MAX(mag)                   AS max_magnitude,
           SUM(tsunami_flag)          AS tsunami_events
    FROM seismic_events
    GROUP BY net, depth_band
    ORDER BY total_events DESC
    LIMIT 10
''').show(truncate=False)

# ═══════════════════════════════════════════════════════════════════════════
# QUERY 2 — Magnitude frequency distribution (Gutenberg-Richter law)
# ═══════════════════════════════════════════════════════════════════════════
print('='*60)
print('QUERY 2 — Magnitude Class Distribution (Gutenberg-Richter)')
print('='*60)
spark.sql('''
    SELECT mag_class,
           COUNT(*) AS event_count,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
    FROM seismic_events
    GROUP BY mag_class
    ORDER BY event_count DESC
''').show()

# ═══════════════════════════════════════════════════════════════════════════
# QUERY 3 — Depth band vs magnitude (tectonic analysis)
# ═══════════════════════════════════════════════════════════════════════════
print('='*60)
print('QUERY 3 — Depth Band vs Magnitude (Tectonic Analysis)')
print('='*60)
spark.sql('''
    SELECT depth_band,
           COUNT(*)                 AS n,
           ROUND(AVG(mag), 3)       AS avg_mag,
           ROUND(AVG(depth_km), 1)  AS avg_depth,
           ROUND(AVG(gap), 1)       AS avg_gap,
           ROUND(AVG(rms), 3)       AS avg_rms,
           MAX(mag)                 AS max_mag
    FROM seismic_events
    GROUP BY depth_band
    ORDER BY avg_mag DESC
''').show()

# ═══════════════════════════════════════════════════════════════════════════
# QUERY 4 — Time-of-day seismic activity pattern
# ═══════════════════════════════════════════════════════════════════════════
print('='*60)
print('QUERY 4 — Hourly Seismic Activity Pattern (UTC)')
print('='*60)
spark.sql('''
    SELECT HOUR(TO_TIMESTAMP(time_iso)) AS hour_of_day,
           COUNT(*)           AS events,
           ROUND(AVG(mag), 3) AS avg_mag,
           MAX(mag)           AS max_mag
    FROM seismic_events
    WHERE time_iso IS NOT NULL
      AND time_iso != ""
    GROUP BY hour_of_day
    ORDER BY hour_of_day
''').show(24)

# ═══════════════════════════════════════════════════════════════════════════
# QUERY 5 — Tsunami risk events by network
# ═══════════════════════════════════════════════════════════════════════════
print('='*60)
print('QUERY 5 — Tsunami Risk Events by Network')
print('='*60)
spark.sql('''
    SELECT net,
           COUNT(*)                 AS tsunami_events,
           ROUND(AVG(mag), 2)       AS avg_mag,
           ROUND(AVG(depth_km), 1)  AS avg_depth,
           MAX(mag)                 AS max_mag
    FROM seismic_events
    WHERE tsunami_flag = 1
    GROUP BY net
    ORDER BY tsunami_events DESC
''').show()

# ═══════════════════════════════════════════════════════════════════════════
# BONUS QUERY 6 — Top 10 most significant earthquakes
# ═══════════════════════════════════════════════════════════════════════════
print('='*60)
print('BONUS QUERY 6 — Top 10 Most Significant Earthquakes')
print('='*60)
spark.sql('''
    SELECT event_id,
           mag,
           depth_km,
           depth_band,
           net,
           sig,
           tsunami_flag,
           time_iso
    FROM seismic_events
    ORDER BY sig DESC
    LIMIT 10
''').show(truncate=50)

# ═══════════════════════════════════════════════════════════════════════════
# BONUS QUERY 7 — Daily earthquake counts (last 30 days trend)
# ═══════════════════════════════════════════════════════════════════════════
print('='*60)
print('BONUS QUERY 7 — Daily Earthquake Trend')
print('='*60)
spark.sql('''
    SELECT DATE(TO_TIMESTAMP(time_iso)) AS event_date,
           COUNT(*)           AS daily_count,
           ROUND(AVG(mag), 3) AS avg_mag,
           MAX(mag)           AS max_mag
    FROM seismic_events
    WHERE time_iso IS NOT NULL
      AND time_iso != ""
    GROUP BY event_date
    ORDER BY event_date DESC
    LIMIT 30
''').show(30)

spark.stop()

print('\n' + '='*60)
print('✅  Week 5c — Spark SQL Analytics COMPLETE!')
print(f'   Events analysed  : {total:,}')
print('   Queries run      : 7 (5 required + 2 bonus)')
print('   Topics covered   :')
print('     • Network hotspot ranking')
print('     • Gutenberg-Richter magnitude distribution')
print('     • Depth band tectonic analysis')
print('     • Hourly activity pattern')
print('     • Tsunami risk by network')
print('     • Most significant earthquakes')
print('     • Daily trend analysis')
print('='*60)