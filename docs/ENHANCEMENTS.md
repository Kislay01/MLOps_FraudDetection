# Enhancements after the faculty review

| Faculty point / pending item | What was added | Where |
|---|---|---|
| Fraud arriving while the model is switching | While a new drift waits for its second confirming reading, the universal model serves (recall-oriented) instead of v1; if the drift is not confirmed, the previous model comes back. `--no-interim` turns it off. | `src/router/policy.py` (`interim_step`), orchestrator |
| New fraud not in our data | A window with no known group alarm but 3+ ungrouped features shifted far beyond their clean-traffic noise is labelled U, so the universal model takes it. `--novel-min 0` turns it off. | `src/router/fingerprint.py` (`classify`) |
| High recall needed | Recall-first thresholds (default target recall 0.90) tuned on holdout data, with a report of the precision cost; served via `FRAUDAR_THRESHOLDS` or pushed live with `POST /admin/threshold`. | `src/zoo/tune_thresholds.py`, API |
| Retraining on a fresh window | After a drift model has served ~2 min, a candidate is trained on the newest drifted rows + clean rows, compared with the serving model on the newest 20% (unseen), registered as `fraud-retrained@candidate`. | `src/retrain/trainer.py` |
| Canary controller | The candidate gets 20% of traffic and is promoted only if its live F2 keeps up; otherwise dropped. Failures never change what is serving. | `src/retrain/controller.py`, `./scripts/demo.sh retraindemo` |
| Alerts | Prometheus rules + Alertmanager → API webhook → control room; also visible in Grafana alerting and the dashboard. | `prometheus/alerts.yml`, `alertmanager/` |
| Cloud deployment + same numbers | One-VM Docker stack, registry bootstrap, parity check of cloud API vs local model files. | `docker-compose.cloud.yml`, `docs/DEPLOY_CLOUD.md`, `src/analysis/parity_check.py` |
| Admin endpoint security | `FRAUDAR_ADMIN_TOKEN` protects all `POST /admin/*` (open when unset, as in the local demo). | `src/settings.py`, API |

## Running the tests

```bash
.venv/bin/pip install -r requirements-dev.txt
./scripts/get_test_tools.sh && source .tools/env.sh   # optional: Prometheus/Alertmanager binaries for the alert tests
.venv/bin/python -m pytest
```

The tests use a small synthetic dataset with the same column names, so they need neither DVC data
nor Kafka. Kafka is replaced by a list of records; the API, orchestrator, retraining controller,
MLflow registry (SQLite), Prometheus and Alertmanager run for real.

## What still needs your machine

- Run `./scripts/demo.sh restart`, then `scenario`, `wrongmodel` and `retraindemo` with the real data.
- Run `python -m src.zoo.tune_thresholds` on the real data and read `reports/threshold_tuning.csv`
  before switching thresholds on.
- `docker compose up -d` now also pulls `prom/alertmanager`.
