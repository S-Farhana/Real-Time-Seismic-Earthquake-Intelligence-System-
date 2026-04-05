import requests
import json
import time
from kafka import KafkaProducer
from datetime import datetime, timezone

KAFKA_BOOTSTRAP = 'localhost:9092'
TOPIC           = 'seismic-alerts'
POLL_INTERVAL   = 60  # seconds

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8')
)

seen_ids = set()

def classify_alert(mag):
    if mag >= 7.0:   return 'red'
    elif mag >= 6.0: return 'orange'
    elif mag >= 5.0: return 'yellow'
    else:            return 'green'

def fetch_and_publish():
    url  = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson'
    try:
        resp = requests.get(url, timeout=10)
        features = resp.json().get('features', [])
        new_count = 0
        for feat in features:
            eid   = feat['id']
            if eid in seen_ids:
                continue
            props = feat['properties']
            coords = feat['geometry']['coordinates']
            mag   = props.get('mag')
            if mag is None:
                continue
            alert = {
                'event_id':    eid,
                'mag':         float(mag),
                'alert_level': classify_alert(float(mag)),
                'place':       props.get('place', ''),
                'time_iso':    datetime.fromtimestamp(
                                   props['time']/1000, tz=timezone.utc
                               ).isoformat() if props.get('time') else '',
                'latitude':    coords[1] if coords else None,
                'longitude':   coords[0] if coords else None,
                'depth_km':    coords[2] if coords else None,
                'tsunami_flag': int(props.get('tsunami', 0)),
                'net':         props.get('net', ''),
                'published_at': datetime.now(timezone.utc).isoformat(),
            }
            producer.send(TOPIC, key=eid, value=alert)
            seen_ids.add(eid)
            new_count += 1
        producer.flush()
        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
        print(f'[{ts}] Alerts published: {new_count}  |  total seen: {len(seen_ids)}')
    except Exception as e:
        print(f'Error: {e}')

print('=' * 55)
print('  SEISMIC ALERT PRODUCER — M4.5+ Events')
print(f'  Topic  : {TOPIC}')
print(f'  Broker : {KAFKA_BOOTSTRAP}')
print('=' * 55)

# Load last hour of M4.5+ events on startup
print('[INIT] Loading recent M4.5+ alerts...')
fetch_and_publish()
print('[INIT] Done. Starting live polling...')

while True:
    time.sleep(POLL_INTERVAL)
    fetch_and_publish()