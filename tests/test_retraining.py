"""Fresh-window retraining and the canary controller."""
import json

import mlflow
import pandas as pd
import pytest
import requests
from mlflow import MlflowClient

import src.api.main as api_mod
from src.drift.drift_generators import apply_drift
from src.retrain import trainer

REAL_LOAD_SLOT = api_mod._load_slot   # captured before any fixture replaces it


@pytest.fixture
def registry(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    MlflowClient(uri).create_experiment("fraudar-retraining", artifact_location=str(tmp_path / "artifacts"))
    monkeypatch.setattr(api_mod, "client", MlflowClient(uri))
    yield uri
    mlflow.set_tracking_uri(api_mod.MLFLOW_TRACKING_URI)


def drift_window(ref, n=6000, seed=60):
    from tests.synth import make_frame
    return apply_drift(make_frame(n, seed=seed), "A", ref, frac=0.3, seed=3)


def test_candidate_beats_stale_model_on_fresh_holdout(base_model):
    from tests.synth import make_frame
    ref = make_frame(5000, seed=61)
    fresh = drift_window(ref)
    train_part, holdout = trainer.split(fresh)
    model, thr, feats = trainer.train_candidate(train_part, make_frame(3000, seed=62), n_estimators=80)
    stale, stale_feats = base_model
    cand = trainer.f2_at(model, thr, holdout, feats)
    old = trainer.f2_at(stale, 0.5, holdout, stale_feats)
    assert cand > old + 0.3, (cand, old)
    assert holdout.index.min() > train_part.index.max()   # holdout is strictly newer


def test_not_enough_fraud_is_refused():
    from tests.synth import make_frame
    clean = make_frame(2000, seed=63, fraud_rate=0.001)
    with pytest.raises(ValueError):
        trainer.train_candidate(clean, pd.DataFrame(), n_estimators=10)


def test_register_and_serve_from_registry(registry, api_client):
    from tests.synth import make_frame
    c, api = api_client
    ref = make_frame(3000, seed=64)
    model, thr, _ = trainer.train_candidate(drift_window(ref, 4000), ref, n_estimators=30)
    v1 = trainer.register(model, thr, {"trained_while_serving": "fraud-spec-A"}, registry)
    v2 = trainer.register(model, thr, {"trained_while_serving": "fraud-spec-A"}, registry)
    assert (v1, v2) == ("1", "2")
    REAL_LOAD_SLOT("fraud-retrained")              # the API's own registry loader
    slot = api.slots["fraud-retrained"]
    assert slot["version"] == "2" and abs(slot["threshold"] - round(thr, 4)) < 1e-9
    r = c.post("/admin/route", json={"model": "fraud-retrained", "weight": 0.2})
    assert r.json()["canary"] == "fraud-retrained"


def test_controller_end_to_end(registry, live_api, monkeypatch, tmp_path, base_model):
    """Live API over HTTP + controller + simulated scoring traffic:
    v1 serves clean rows, the orchestrator routes drift A to spec-A, the controller retrains on the
    fresh window, canaries the candidate and promotes it once its live F2 beats spec-A's."""
    from tests.synth import make_frame
    import src.retrain.controller as ctl
    url, api = live_api
    fake = api._load_slot
    monkeypatch.setattr(api, "_load_slot", lambda n: REAL_LOAD_SLOT(n) if n == "fraud-retrained" else fake(n))
    monkeypatch.setattr(ctl, "LOG", str(tmp_path / "retrain.jsonl"))

    ref = make_frame(5000, seed=70)
    clean = make_frame(3000, seed=71).assign(injected=0)
    drift = drift_window(ref, 9000, seed=72)
    recs = json.loads(pd.concat([clean, drift]).to_json(orient="records"))
    s = requests.Session()

    def stream():
        for i, r in enumerate(recs):
            if i == len(clean):   # what the orchestrator would do once drift A is confirmed
                s.post(f"{url}/admin/route", json={"model": "fraud-spec-A", "reason": "fingerprint=A"})
            s.post(f"{url}/predict", json={"data": r, "label": r["isFraud"]})   # scoring consumer
            yield type("M", (), {"value": r})

    stale = base_model[0]
    c = ctl.main(["--api", url, "--min-rows", "3000", "--clean-rows", "3000", "--trees", "80",
                  "--metrics-port", "0", "--cooldown-rows", "100000"],
                 consumer=stream(),
                 register=lambda m, t, tags: trainer.register(m, t, tags, registry),
                 load_serving=lambda name, alias: (stale, 0.5, list(stale.get_booster().feature_names)))
    events = [json.loads(x)["event"] for x in (tmp_path / "retrain.jsonl").read_text().splitlines()]
    assert events == ["canary_started", "promoted"], events
    assert api.state["active"] == "fraud-retrained" and api.state["canary"] is None
    assert [h["event"] for h in api.history][-4:] == ["cutover", "reload", "canary", "cutover"]
    assert c.phase == "idle"


def test_registry_down_does_not_crash(live_api, monkeypatch, tmp_path, base_model):
    """If MLflow is unreachable the controller logs deploy_failed and the API keeps serving."""
    from tests.synth import make_frame
    import src.retrain.controller as ctl
    url, api = live_api
    monkeypatch.setattr(ctl, "LOG", str(tmp_path / "retrain.jsonl"))
    ref = make_frame(5000, seed=80)
    recs = json.loads(drift_window(ref, 4000, seed=81).to_json(orient="records"))
    requests.post(f"{url}/admin/route", json={"model": "fraud-spec-A"})

    def boom(*a):
        raise ConnectionError("registry unreachable")

    stale = base_model[0]
    c = ctl.main(["--api", url, "--min-rows", "2000", "--trees", "30", "--metrics-port", "0"],
                 consumer=(type("M", (), {"value": r}) for r in recs), register=boom,
                 load_serving=lambda n, a: (stale, 0.5, list(stale.get_booster().feature_names)))
    events = [json.loads(x)["event"] for x in (tmp_path / "retrain.jsonl").read_text().splitlines()]
    assert events == ["deploy_failed"]
    assert api.state["active"] == "fraud-spec-A" and api.state["canary"] is None
    assert c.phase == "collecting"
