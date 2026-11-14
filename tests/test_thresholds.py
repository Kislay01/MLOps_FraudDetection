"""Recall-first threshold tuning and the live threshold endpoint."""
import json

import joblib
import numpy as np

from src.drift.drift_generators import apply_drift
from src.zoo.tune_thresholds import main as tune, recall_threshold


def test_recall_threshold_reaches_target():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(0.3 * y + rng.random(5000) * 0.7, 0, 1)
    for target in (0.8, 0.9, 0.95):
        t = recall_threshold(y, p, target)
        assert ((p >= t) & (y == 1)).sum() / y.sum() >= target


def test_tune_script_end_to_end(tmp_path):
    from tests.synth import fit, make_frame
    data, models, reports = tmp_path / "ds", tmp_path / "m", tmp_path / "r"
    data.mkdir(); models.mkdir()
    ref = make_frame(6000, seed=41)
    val = make_frame(34000, seed=42)
    clean_test = make_frame(4000, seed=43)
    m, feats = fit(ref)
    joblib.dump(m, models / "xgb_fraud_model_v1.pkl")
    for k in ["A", "B", "C", "U"]:
        tr = apply_drift(make_frame(5000, seed=50), k, ref, frac=0.1, seed=1)
        joblib.dump(tr, data / f"{k}_train.pkl")
        joblib.dump(apply_drift(clean_test, k, ref, frac=0.1, seed=2), data / f"{k}_test.pkl")
        mk, _ = fit(tr)
        name = "universal" if k == "U" else f"spec_{k}"
        joblib.dump({"model": mk, "features": feats, "threshold": 0.5}, models / f"zoo_{name}.pkl")
    joblib.dump(clean_test, data / "normal_test.pkl")
    val_path = tmp_path / "val.pkl"
    joblib.dump(val, val_path)

    new, report = tune(["--data-dir", str(data), "--val", str(val_path), "--models-dir", str(models),
                        "--reports-dir", str(reports), "--target-recall", "0.9"])
    saved = json.loads((models / "thresholds_recall.json").read_text())["thresholds"]
    assert set(saved) == {"fraud-detection-model", "fraud-spec-A", "fraud-spec-B", "fraud-spec-C", "fraud-universal"}
    assert all(0 < t <= 0.6210 for t in saved.values())
    rf = report[report.threshold_kind == "recall_first"].set_index(["model", "test_set"]).recall
    f2o = report[report.threshold_kind == "f2_optimal"].set_index(["model", "test_set"]).recall
    assert (rf >= f2o - 1e-9).all()          # recall never drops versus the F2-optimal threshold
    assert (reports / "threshold_tuning.csv").exists()


def test_live_threshold_endpoint(api_client, monkeypatch):
    c, api = api_client
    r = c.post("/admin/threshold", json={"model": "fraud-spec-A", "threshold": 0.12})
    assert r.status_code == 200 and api.slots["fraud-spec-A"]["threshold"] == 0.12
    assert c.post("/admin/threshold", json={"model": "fraud-spec-A", "threshold": 1.5}).status_code == 400
    assert c.post("/admin/threshold", json={"model": "nope", "threshold": 0.2}).status_code == 400
    monkeypatch.setattr(api, "ADMIN_TOKEN", "t")
    assert c.post("/admin/threshold", json={"model": "fraud-spec-A", "threshold": 0.2}).status_code == 401


def test_threshold_file_override(monkeypatch, tmp_path):
    import src.api.main as api
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"thresholds": {"fraud-spec-B": 0.07}}))
    monkeypatch.setattr(api, "THRESHOLDS_FILE", str(f))
    assert api._threshold_override("fraud-spec-B") == 0.07
    assert api._threshold_override("fraud-spec-A") is None
    monkeypatch.setattr(api, "THRESHOLDS_FILE", "")
    assert api._threshold_override("fraud-spec-B") is None
