import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import requests, json, time
from kafka import KafkaProducer
from datetime import datetime
from config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, TOPICS, POLL_INTERVAL_SEC
from config.api_config import USGS_HOUR_FEED, USGS_MONTH_FEED

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8')
)

seen_ids = set()

def extract_event(feature):
    props  = feature['properties']
    coords = feature['geometry']['coordinates']
    return {
        'event_id':    feature['id'],
        'mag':         props.get('mag'),
        'place':       props.get('place'),
        'time_ms':     props.get('time'),
        'time_iso':    datetime.utcfromtimestamp(
                           props['time'] / 1000).isoformat()
                           if props.get('time') else None,
        'updated_ms':  props.get('updated'),
        'longitude':   coords[0] if coords else None,
        'latitude':    coords[1] if coords else None,
        'depth_km':    coords[2] if coords else None,
        'gap':         props.get('gap'),
        'rms':         props.get('rms'),
        'nst':         props.get('nst'),
        'dmin':        props.get('dmin'),
        'sig':         props.get('sig'),
        'net':         props.get('net'),
        'mag_type':    props.get('magType'),
        'event_type':  props.get('type'),
        'alert':       props.get('alert'),
        'tsunami':     props.get('tsunami'),
        'felt':        props.get('felt'),
        'status':      props.get('status'),
        'ingested_at': datetime.utcnow().isoformat()
    }

def publish_feed(feed_url, label=''):
    resp     = requests.get(feed_url, timeout=30)
    features = resp.json()['features']
    count    = 0
    for feat in features:
        eid = feat['id']
        if eid not in seen_ids:
            event = extract_event(feat)
            producer.send(TOPICS['events'], key=eid, value=event)
            seen_ids.add(eid)
            count += 1
    producer.flush()
    ts = datetime.utcnow().strftime('%H:%M:%S')
    print(f'[{ts}] {label} +{count} events  |  total seen: {len(seen_ids)}')

print("=" * 55)
print("  SEISMIC KAFKA PRODUCER — LIVE")
print(f"  Topic  : {TOPICS['events']}")
print(f"  Broker : {KAFKA_BOOTSTRAP_SERVERS}")
print("=" * 55)

# Step 1 — load 30 days history for ML training
print('\n[INIT] Loading 30-day history for ML training...')
publish_feed(USGS_MONTH_FEED, label='[HISTORY]')
print('[INIT] History loaded. Starting live polling...\n')

# Step 2 — poll live feed every 60 seconds forever
while True:
    try:
        publish_feed(USGS_HOUR_FEED, label='[LIVE]   ')
    except Exception as e:
        print(f'[ERROR] {e}')
    time.sleep(POLL_INTERVAL_SEC)