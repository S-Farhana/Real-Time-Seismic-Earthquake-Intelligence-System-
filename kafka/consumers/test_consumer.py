import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from kafka import KafkaConsumer
import json
from config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, TOPICS

consumer = KafkaConsumer(
    TOPICS['events'],
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("=" * 55)
print("  SEISMIC CONSUMER — LISTENING")
print(f"  Topic : {TOPICS['events']}")
print("=" * 55 + "\n")

for i, msg in enumerate(consumer):
    e     = msg.value
    mag   = e.get('mag')
    place = e.get('place', 'Unknown')
    depth = e.get('depth_km')
    ts    = e.get('time_iso', '')[:19]

    flag = ""
    if mag and mag >= 5.0: flag = "  *** SIGNIFICANT ***"
    if mag and mag >= 7.0: flag = "  !!! MAJOR EVENT !!!"

    print(f"Event #{i+1}{flag}")
    print(f"  Location  : {place}")
    print(f"  Magnitude : {mag}")
    print(f"  Depth     : {depth} km")
    print(f"  Time (UTC): {ts}")
    print(f"  Network   : {e.get('net')}  |  ID: {e.get('event_id')}")
    print()

    if i >= 14:
        print("--- showing first 15 events, stopping ---")
        break
