"""Service addresses and the admin token, read from environment variables.
Defaults match the local demo, so nothing needs to be set when running on one machine."""
import os

KAFKA = os.getenv("FRAUDAR_KAFKA", "localhost:9092")
MLFLOW_URI = os.getenv("FRAUDAR_MLFLOW", "http://localhost:5000")
API_URL = os.getenv("FRAUDAR_API", "http://localhost:8000")
# When set, POST /admin/* on the API requires the header X-Admin-Token with this value.
ADMIN_TOKEN = os.getenv("FRAUDAR_ADMIN_TOKEN", "")


def admin_headers():
    return {"X-Admin-Token": ADMIN_TOKEN} if ADMIN_TOKEN else {}
