import requests
import json
import time
import math
import random
from kafka import KafkaProducer
from datetime import datetime, timezone

KAFKA_BOOTSTRAP = 'localhost:9092'
TOPIC           = 'seismic-waveforms'
POLL_INTERVAL   = 60  # seconds

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8')
)

seen_ids = set()

# Seismic station network (real USGS stations)
STATIONS = [
    {'station_id': 'IU.ANMO.00', 'name': 'Albuquerque, NM',   'lat': 34.9459, 'lon': -106.4572},
    {'station_id': 'IU.COLA.00', 'name': 'College, Alaska',    'lat': 64.8736, 'lon': -147.8616},
    {'station_id': 'IU.HRV.00',  'name': 'Harvard, MA',        'lat': 42.5064, 'lon': -71.5583},
    {'station_id': 'IU.KMBO.00', 'name': 'Kilima Mbogo, Kenya','lat': -1.1271, 'lon':  37.2527},
    {'station_id': 'IU.MAJO.00', 'name': 'Matsushiro, Japan',  'lat': 36.5457, 'lon': 138.2041},
]

def generate_waveform(mag, depth_km, distance_km, n_samples=100):
    """Generate synthetic seismic waveform amplitudes."""
    amplitude_base = 10 ** (mag - 3) / max(distance_km, 1)
    waveform = []
    for i in range(n_samples):
        t = i / 100.0
        # P-wave + S-wave simulation
        p_wave = amplitude_base * math.sin(2 * math.pi * 2 * t) * math.exp(-t * 0.5)
        s_wave = amplitude_base * 1.7 * math.sin(2 * math.pi * 1 * t) * math.exp(-(t-2) * 0.3) if t > 0.5 else 0
        noise  = random.gauss(0, amplitude_base * 0.05)
        waveform.append(round(p_wave + s_wave + noise, 6))
    return waveform

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def fetch_and_publish():
    url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson'
    try:
        resp     = requests.get(url, timeout=10)
        features = resp.json().get('features', [])
        new_count = 0

        for feat in features:
            eid = feat['id']
            if eid in seen_ids:
                continue

            props  = feat['properties']
            coords = feat['geometry']['coordinates']
            mag    = props.get('mag')

            if mag is None or coords is None:
                continue

            eq_lat   = float(coords[1])
            eq_lon   = float(coords[0])
            eq_depth = float(coords[2])

            # Publish waveform reading from each station
            for station in STATIONS:
                dist_km = haversine(eq_lat, eq_lon,
                                    station['lat'], station['lon'])

                # Only publish if station is within 10,000 km
                if dist_km > 10000:
                    continue

                waveform = generate_waveform(float(mag), eq_depth, dist_km)
                snr      = round(10 * math.log10(max(abs(max(waveform)), 0.001) /
                                                  max(abs(min(waveform)), 0.001) + 1), 2)

                record = {
                    'event_id':      eid,
                    'station_id':    station['station_id'],
                    'station_name':  station['name'],
                    'timestamp_ms':  props.get('time', 0),
                    'time_iso':      datetime.fromtimestamp(
                                         props['time']/1000, tz=timezone.utc
                                     ).isoformat() if props.get('time') else '',
                    'eq_mag':        float(mag),
                    'eq_lat':        eq_lat,
                    'eq_lon':        eq_lon,
                    'eq_depth_km':   eq_depth,
                    'station_lat':   station['lat'],
                    'station_lon':   station['lon'],
                    'distance_km':   round(dist_km, 2),
                    'channel_bhz':   waveform,          # Vertical component
                    'channel_bhn':   [w * 0.7 + random.gauss(0, 0.01) for w in waveform],  # N-S
                    'channel_bhe':   [w * 0.6 + random.gauss(0, 0.01) for w in waveform],  # E-W
                    'sampling_rate': 100,
                    'units':         'nm/s',
                    'snr':           snr,
                    'net':           props.get('net', 'us'),
                    'ingested_at':   datetime.now(timezone.utc).isoformat(),
                }

                key = f'{eid}#{station["station_id"]}'
                producer.send(TOPIC, key=key, value=record)
                new_count += 1

            seen_ids.add(eid)

        producer.flush()
        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
        print(f'[{ts}] Waveform records published: {new_count}  |  events seen: {len(seen_ids)}')

    except Exception as e:
        print(f'Error: {e}')

print('=' * 55)
print('  SEISMIC WAVEFORM PRODUCER')
print(f'  Topic    : {TOPIC}')
print(f'  Broker   : {KAFKA_BOOTSTRAP}')
print(f'  Stations : {len(STATIONS)} global IRIS stations')
print('=' * 55)

print('[INIT] Loading recent M2.5+ events...')
fetch_and_publish()
print('[INIT] Done. Starting live polling...')

while True:
    time.sleep(POLL_INTERVAL)
    fetch_and_publish()