"""
Kafka producer that replays historical transaction data as a live stream,
simulating real-time transaction arrivals for the fraud detection pipeline.
"""

import json
import time
import argparse
import joblib
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime, timedelta, timezone

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC_NAME = 'transactions'


def load_data(data_path):
    """Load processed data, sorted chronologically by TransactionDT."""
    df = joblib.load(data_path)
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    return df


def anchor_time(df, anchor_datetime=None):
    """
    TransactionDT is a relative delta (seconds from an arbitrary reference point),
    not a real timestamp. We anchor the first transaction to a real wall-clock time
    so downstream consumers can reason about actual elapsed time.
    """
    if anchor_datetime is None:
        anchor_datetime = datetime.now(timezone.utc)

    min_dt = df['TransactionDT'].min()
    df = df.copy()
    df['real_timestamp'] = df['TransactionDT'].apply(
        lambda x: anchor_datetime + timedelta(seconds=int(x - min_dt))
    )
    return df


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
        key_serializer=lambda k: str(k).encode('utf-8') if k is not None else None,
    )


def replay(df, producer, speed_multiplier=1000, max_rows=None):
    """
    Replay transactions onto Kafka, pacing them according to their real time gaps,
    compressed by speed_multiplier (e.g. 1000x means 1000 seconds of real data
    plays out in 1 second of wall-clock time).
    """
    if max_rows:
        df = df.head(max_rows)

    total = len(df)
    print(f"Starting replay of {total} transactions at {speed_multiplier}x speed...")

    last_dt = None
    sent = 0

    for _, row in df.iterrows():
        current_dt = row['TransactionDT']

        if last_dt is not None:
            gap_seconds = current_dt - last_dt
            sleep_time = gap_seconds / speed_multiplier
            if sleep_time > 0:
                time.sleep(sleep_time)

        record = row.drop(labels=['real_timestamp']).to_dict()

        producer.send(
            TOPIC_NAME,
            key=row['TransactionID'],
            value=record
        )

        sent += 1
        last_dt = current_dt

        if sent % 1000 == 0:
            print(f"Sent {sent}/{total} transactions...")

    producer.flush()
    print(f"Replay complete. Sent {sent} transactions total.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay historical transactions to Kafka')
    parser.add_argument('--data', type=str, default='data/processed/train_processed_v2.pkl',
                        help='Path to processed data pickle file')
    parser.add_argument('--speed', type=float, default=1000,
                        help='Speed multiplier (higher = faster replay)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Limit number of rows to replay (for testing)')
    args = parser.parse_args()

    df = load_data(args.data)
    df = anchor_time(df)

    producer = create_producer()
    replay(df, producer, speed_multiplier=args.speed, max_rows=args.max_rows)