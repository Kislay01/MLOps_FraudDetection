"""Cloud deployment pieces: registry bootstrap, API start-up from a registry, parity check, compose files."""
import json
import os
import shutil
import subprocess

import joblib
import mlflow
import pytest
import yaml
from mlflow import MlflowClient

import src.api.main as api_mod
from src.drift.drift_generators import apply_drift
from src.deploy import bootstrap_registry

REAL_LOAD_SLOT = api_mod._load_slot


@pytest.fixture
def model_files(tmp_path, base_model, frame):
    from tests.synth import fit, make_frame
    root = tmp_path / "root"
    (root / "models").mkdir(parents=True)
    (root / "ds").mkdir()
    m, feats = base_model
    joblib.dump(m, root / "models/xgb_fraud_model_v1.pkl")
    ref = make_frame(4000, seed=90)
    test = make_frame(3000, seed=91)
    joblib.dump(test, root / "ds/normal_test.pkl")
    for k in ["A", "B", "C", "U"]:
        d = apply_drift(make_frame(3000, seed=92), k, ref, frac=0.1, seed=1)
        mk, _ = fit(d, n_estimators=30)
        name = "universal" if k == "U" else f"spec_{k}"
        joblib.dump({"model": mk, "features": feats, "threshold": 0.4}, root / f"models/zoo_{name}.pkl")
        joblib.dump(apply_drift(test, k, ref, frac=0.1, seed=2), root / f"ds/{k}_test.pkl")
    return root


@pytest.fixture
def registry(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    MlflowClient(uri).create_experiment("fraudar-bootstrap", artifact_location=str(tmp_path / "artifacts"))
    monkeypatch.setattr(api_mod, "client", MlflowClient(uri))
    yield uri
    mlflow.set_tracking_uri(api_mod.MLFLOW_TRACKING_URI)


def test_bootstrap_then_api_starts_from_registry(model_files, registry, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    bootstrap_registry.main(registry, str(model_files))
    bootstrap_registry.main(registry, str(model_files))          # idempotent
    c = MlflowClient(registry)
    for name in ["fraud-detection-model", "fraud-spec-A", "fraud-spec-B", "fraud-spec-C", "fraud-universal"]:
        assert len(c.search_model_versions(f"name='{name}'")) == 1
    assert c.get_model_version_by_alias("fraud-detection-model", "champion").tags["threshold"] == "0.621"

    api_mod.slots.clear(); api_mod.history.clear(); api_mod.perf.clear()
    api_mod.state.update(active=api_mod.DEFAULT_MODEL, previous=None, canary=None, canary_weight=0.0)
    monkeypatch.setattr(api_mod, "_load_slot", REAL_LOAD_SLOT)
    monkeypatch.setattr(api_mod, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    with TestClient(api_mod.app) as cl:          # the real start-up path, models loaded from MLflow
        h = cl.get("/health").json()
        assert h["loaded_models"] == ["fraud-detection-model", "fraud-spec-A", "fraud-spec-B",
                                      "fraud-spec-C", "fraud-universal"]
        rec = joblib.load(model_files / "ds/normal_test.pkl").iloc[0].to_dict()
        assert cl.post("/predict", json={"data": rec}).status_code == 200


def test_parity_check_against_live_api(model_files, live_api):
    from src.analysis.parity_check import main as parity
    url, api = live_api
    # the live API serves base_model in every slot, so only v1 matches its local file exactly
    ok, res = parity(["--api", url, "--rows", "800", "--data-dir", str(model_files / "ds"),
                      "--models-dir", str(model_files / "models"), "--models", "fraud-detection-model"])
    assert ok and res[0]["max_abs_diff"] == 0.0 and res[0]["f2_local"] == res[0]["f2_api"]
    ok, res = parity(["--api", url, "--rows", "800", "--data-dir", str(model_files / "ds"),
                      "--models-dir", str(model_files / "models"), "--models", "fraud-spec-A"])
    assert not ok and res[0]["max_abs_diff"] > 1e-3      # a different model is caught


def test_score_endpoint_has_no_side_effects(api_client, frame):
    c, api = api_client
    rows = json.loads(frame.iloc[:50].to_json(orient="records"))
    before = (dict(api.state), len(api.history), {k: len(v) for k, v in api.perf.items()})
    r = c.post("/admin/score", json={"model": "fraud-spec-B", "rows": rows})
    assert r.status_code == 200 and len(r.json()["probabilities"]) == 50
    assert before == (dict(api.state), len(api.history), {k: len(v) for k, v in api.perf.items()})


def test_compose_files_are_consistent():
    cloud = yaml.safe_load(open("docker-compose.cloud.yml"))
    s = cloud["services"]
    assert {"kafka", "mlflow", "bootstrap", "api", "consumer", "orchestrator", "retrain", "producer",
            "prometheus", "alertmanager", "grafana"} <= set(s)
    prom = yaml.safe_load(open("deploy/prometheus.cloud.yml"))
    targets = {t for j in prom["scrape_configs"] for sc in j["static_configs"] for t in sc["targets"]}
    for t in targets:
        assert t.split(":")[0] in s, t
    assert s["api"]["ports"] == ["127.0.0.1:8000:8000"]           # API not public
    for name in ("api", "orchestrator", "retrain", "consumer", "producer"):
        assert s[name]["environment"]["FRAUDAR_KAFKA"] == "kafka:29092"
    am = yaml.safe_load(open("deploy/alertmanager.cloud.yml"))
    assert am["receivers"][0]["webhook_configs"][0]["url"] == "http://api:8000/alerts/webhook"
    local = yaml.safe_load(open("docker-compose.yml"))
    assert "alertmanager" in local["services"]


@pytest.mark.skipif(not shutil.which("docker"), reason="docker CLI not available")
def test_compose_config_validates(tmp_path):
    env = dict(os.environ, GRAFANA_PASSWORD="x")
    created = not os.path.exists(".env")
    if created:
        shutil.copy(".env.example", ".env")
    try:
        for f in ("docker-compose.yml", "docker-compose.cloud.yml"):
            r = subprocess.run(["docker", "compose", "-f", f, "config", "-q"], capture_output=True, text=True, env=env)
            assert r.returncode == 0, r.stderr
    finally:
        if created:
            os.remove(".env")
