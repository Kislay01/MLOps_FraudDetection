import json
import joblib
import pandas as pd
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC_NAME = 'transactions'
MODEL_PATH = 'models/xgb_fraud_model_v1.pkl'
CONSUMER_GROUP = 'fraud-detection-consumer'

EXCLUDE_COLS = ['TransactionID', 'isFraud', 'TransactionDT', 'card_entity']


def load_model():
    return joblib.load(MODEL_PATH)


def create_consumer():
    return KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        group_id=CONSUMER_GROUP,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        key_deserializer=lambda k: k.decode('utf-8') if k else None,
    )


def predict_single(model, record, feature_cols):
    """Build a single-row DataFrame matching training feature order, run inference."""
    row = {col: record.get(col, 0) for col in feature_cols}
    df = pd.DataFrame([row])
    proba = model.predict_proba(df)[0, 1]
    return proba


def main():
    print("Loading model...")
    model = load_model()
    feature_cols = model.get_booster().feature_names

    print(f"Model loaded. Expecting {len(feature_cols)} features.")
    print("Connecting to Kafka...")

    consumer = create_consumer()

    print(f"Listening on topic '{TOPIC_NAME}'. Waiting for messages...\n")

    processed = 0
    flagged = 0

    for message in consumer:
        record = message.value
        txn_id = record.get('TransactionID')
        actual_fraud = record.get('isFraud')

        proba = predict_single(model, record, feature_cols)
        predicted_fraud = 1 if proba >= 0.6210 else 0

        processed += 1
        if predicted_fraud == 1:
            flagged += 1

        status = "FLAGGED" if predicted_fraud == 1 else "clear"
        match = "✓" if predicted_fraud == actual_fraud else "✗"

        print(f"[{processed}] TxnID={txn_id} | fraud_prob={proba:.4f} | {status} | actual={actual_fraud} {match}")

        if processed % 100 == 0:
            print(f"\n--- Processed {processed} transactions, {flagged} flagged as fraud ---\n")


if __name__ == '__main__':
    main()