"""Switch-gap cover (interim universal model) and unknown-drift fallback."""
import json
import sys

import pandas as pd

from src.drift.drift_generators import apply_drift
from src.router.fingerprint import classify
from src.router.policy import ROUTES, RoutingPolicy, interim_step


def run(labels, interim_on=True):
    p, interim, routes = RoutingPolicy(), False, []
    route = ROUTES["normal"]
    for lab in labels:
        d = p.update(lab)
        if d["event"] in ("switch", "recover"):
            route = d["route"]
        if interim_on:
            to, interim, _ = interim_step(d["event"], p.state, interim)
            route = to or route
        routes.append(route)
    return routes


def test_interim_covers_confirmation_gap():
    r = run(["normal", "A", "A", "A", "normal", "normal", "normal"])
    assert r == ["fraud-detection-model", "fraud-universal", "fraud-spec-A", "fraud-spec-A",
                 "fraud-spec-A", "fraud-spec-A", "fraud-detection-model"]
    # without interim the first drifted window stays on v1 (the old behaviour)
    assert run(["normal", "A", "A"], interim_on=False)[1] == "fraud-detection-model"


def test_interim_ends_when_drift_not_confirmed():
    assert run(["A", "normal"]) == ["fraud-universal", "fraud-detection-model"]


def test_interim_between_two_drifts():
    r = run(["A", "A", "C", "C"])
    assert r == ["fraud-universal", "fraud-spec-A", "fraud-universal", "fraud-spec-C"]


def test_no_interim_when_already_universal():
    assert run(["U", "U", "A", "A"]) == ["fraud-universal", "fraud-universal", "fraud-universal", "fraud-spec-A"]


def _windows(fp, df, w=1000, step=500):
    return [fp.score(df.iloc[i:i + w]) for i in range(0, len(df) - w + 1, step)]


def test_clean_traffic_never_flagged(calibrated_fp):
    from tests.synth import make_frame
    fp, ref = calibrated_fp
    fp.set_reference(ref.iloc[-5000:])
    for s in _windows(fp, make_frame(8000, seed=21)):
        assert classify(s, novel_min=3) == "normal", s


def test_known_drifts_keep_their_labels(calibrated_fp):
    from tests.synth import make_frame
    fp, ref = calibrated_fp
    fp.set_reference(ref.iloc[-5000:])
    clean = make_frame(3000, seed=22)
    for k in ["A", "B", "C"]:
        s = fp.score(apply_drift(clean, k, ref, frac=0.3, seed=1).iloc[:1000])
        assert classify(s) == k
        assert classify(s, novel_min=3) == k


def test_unknown_drift_goes_to_universal(calibrated_fp):
    """A shift on a few features outside every group: old classifier sees 'normal', new one says U."""
    from tests.synth import make_frame
    fp, ref = calibrated_fp
    fp.set_reference(ref.iloc[-5000:])
    w = make_frame(1000, seed=23)
    idx = w.sample(frac=0.3, random_state=0).index
    for c in ["D3", "D4", "D5", "D6", "V40"]:
        w.loc[idx, c] = w.loc[idx, c] * 6 + 50
    s = fp.score(w)
    assert s["breadth"] < 20
    assert classify(s) == "normal"
    assert classify(s, novel_min=3) == "U"


def test_orchestrator_end_to_end(calibrated_fp, live_api, monkeypatch, tmp_path):
    """Real orchestrator loop + real API over HTTP, Kafka replaced by a list of records:
    clean -> drift A -> clean must give interim universal, then specialist A, then v1."""
    from tests.conftest import FakeConsumer
    from tests.synth import make_frame
    import src.orchestrator.orchestrator as orch
    fp, ref = calibrated_fp
    url, api = live_api
    clean = make_frame(17000, seed=31)
    drifted = apply_drift(clean.iloc[8000:12000], "A", ref, frac=0.3, seed=2)
    stream = pd.concat([clean.iloc[:8000].assign(injected=0), drifted,
                        clean.iloc[12000:].assign(injected=0)])
    recs = json.loads(stream.to_json(orient="records"))
    log = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(orch, "KafkaConsumer", FakeConsumer(recs))
    monkeypatch.setattr(orch.joblib, "load", lambda p: fp)
    monkeypatch.setattr(orch, "start_http_server", lambda port: None)
    monkeypatch.setattr(orch, "DECISIONS_LOG", str(log))
    monkeypatch.setattr(sys, "argv", ["orch", "--api", url])
    orch.main()

    rows = [json.loads(x) for x in log.read_text().splitlines()]
    events = [(r["rows_seen"], r["event"], r["interim"], r["route"]) for r in rows
              if r["event"] in ("switch", "recover") or r["interim"]]
    kinds = [e[1] if e[1] != "pending" else e[2] for e in events]
    assert kinds == ["interim", "switch", "recover"], events
    assert events[1][3] == "fraud-spec-A" and events[2][3] == "fraud-detection-model"
    assert all(r["label"] == "normal" for r in rows if r["rows_seen"] <= 8000)
    hist = [(h["event"], h["model"]) for h in api.history]
    assert hist[-3:] == [("cutover", "fraud-universal"), ("cutover", "fraud-spec-A"),
                         ("cutover", "fraud-detection-model")]
