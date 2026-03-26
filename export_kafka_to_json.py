from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'seismic-events',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    consumer_timeout_ms=10000,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

events = []
for msg in consumer:
    events.append(msg.value)

print(f'Exported {len(events)} events')

with open('seismic_events.json', 'w') as f:
    for event in events:
        f.write(json.dumps(event) + '\n')

print('Saved to seismic_events.json')