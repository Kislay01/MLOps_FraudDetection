# Deploying Fraudar on a cloud VM

The whole stack runs on one Linux VM with Docker: Kafka, MLflow, the API, scoring consumer,
orchestrator, retraining controller, stream producer, Prometheus, Alertmanager and Grafana.
Any provider works (AWS EC2, GCP Compute Engine, Azure VM, Oracle Cloud). The local demo
(`./scripts/demo.sh`) is unchanged.

## 1. VM

- Ubuntu 22.04 or 24.04, 4 vCPU, 16 GB RAM, 40 GB disk (e.g. AWS t3.xlarge, GCP e2-standard-4).
- Firewall / security group: allow inbound **22** (SSH) and **3000** (Grafana) only.
  The API is bound to `127.0.0.1` on the VM; reach it through an SSH tunnel.
- Install Docker: `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER` (log out and in).

## 2. Code, data and models

```bash
git clone https://github.com/Kislay01/MLOps_FraudDetection.git && cd MLOps_FraudDetection
git checkout faculty-enhancements
python3 -m venv .venv && .venv/bin/pip install "dvc[s3]"
# R2 credentials (same as on your laptop)
.venv/bin/dvc remote modify --local r2remote access_key_id <KEY>
.venv/bin/dvc remote modify --local r2remote secret_access_key <SECRET>
.venv/bin/dvc pull
```

## 3. Configure and start

```bash
cp .env.example .env
# edit .env: set FRAUDAR_ADMIN_TOKEN to a long random string (openssl rand -hex 32) and GRAFANA_PASSWORD
```

Pin model libraries to the versions that trained the models. On your laptop:
`.venv/bin/pip freeze | grep -iE "^(xgboost|scikit-learn)=="`, then put those exact lines in
`requirements.txt` on the VM. Then:

```bash
docker compose -f docker-compose.cloud.yml up -d --build
docker compose -f docker-compose.cloud.yml logs -f orchestrator   # wait for "warm-up complete" (~100 s)
```

`bootstrap` registers v1 and the model zoo from `models/*.pkl` into the VM's MLflow once
(re-running it skips models that are already registered).

## 4. Use it

- Dashboard: `http://<vm-ip>:3000/d/fraudar-routing` (user `admin`, password from `.env`).
- Inject a drift: `docker compose -f docker-compose.cloud.yml exec producer python -m src.streaming.inject --type A`
- Clear it: `... exec producer python -m src.streaming.inject --clear`
- Full scenario: `... exec producer python -m src.streaming.inject --scenario demo --hold 45 --gap 40`
- Alerts: Grafana → Alerting, or `curl localhost:8000/alerts` on the VM.

## 5. Check the deployed model keeps its numbers

From your laptop (it needs the local `data/` and `models/`):

```bash
ssh -N -L 8000:localhost:8000 <user>@<vm-ip> &
FRAUDAR_ADMIN_TOKEN=<token from .env> .venv/bin/python -m src.analysis.parity_check --api http://localhost:8000
```

It scores the same test rows with the VM's API and with your local model files and prints the
largest probability difference and F2 both ways for every model. `PARITY OK` means the cloud
deployment reproduces the local numbers; a failure usually means different xgboost/scikit-learn
versions (fix the pins in step 3).

## 6. Stop / clean up

```bash
docker compose -f docker-compose.cloud.yml down        # keep volumes (MLflow, Grafana)
docker compose -f docker-compose.cloud.yml down -v     # remove everything
```

Stop or delete the VM when you are not using it to avoid charges.
