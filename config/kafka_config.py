KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'

TOPICS = {
    'events':    'seismic-events',
    'waveforms': 'seismic-waveforms',
    'alerts':    'seismic-alerts',
}

CONSUMER_GROUP    = 'seismic-consumer-group'
POLL_INTERVAL_SEC = 60