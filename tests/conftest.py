import os
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


@pytest.fixture(scope="session")
def frame():
    from tests.synth import make_frame
    return make_frame()


@pytest.fixture(scope="session")
def base_model(frame):
    from tests.synth import fit
    return fit(frame)


def _fake_loader(api, model):
    def load(name):
        api.slots[name] = {"model": model, "version": "1", "alias": api.SLOT_ALIASES[name],
                           "threshold": 0.5, "kind": "test",
                           "feature_names": api._single_thread(model)}
    return load


@pytest.fixture
def api_client(monkeypatch, base_model):
    """The real FastAPI app with registry loading replaced by an in-memory model."""
    from fastapi.testclient import TestClient
    import src.api.main as api
    api.slots.clear()
    api.history.clear()
    api.perf.clear()
    api.state.update(active=api.DEFAULT_MODEL, previous=None, canary=None, canary_weight=0.0)
    monkeypatch.setattr(api, "_load_slot", _fake_loader(api, base_model[0]))
    with TestClient(api.app) as c:
        yield c, api


@pytest.fixture(scope="session")
def calibrated_fp():
    """Fingerprinter calibrated on synthetic clean traffic, like calibrate_fingerprint.py does."""
    from tests.synth import make_frame
    from src.router.fingerprint import Fingerprinter
    ref = make_frame(20000, seed=11)
    fp = Fingerprinter(ref.iloc[:5000])
    fp.calibrate_rolling(ref, list(range(5000, 19000, 1000)), 5000, 1000)
    return fp, ref


@pytest.fixture
def live_api(monkeypatch, base_model, tmp_path):
    """The real API served over HTTP on a free port (registry loading replaced in memory)."""
    import socket
    import threading
    import time
    import uvicorn
    import src.api.main as api
    api.slots.clear(); api.history.clear(); api.perf.clear()
    api.state.update(active=api.DEFAULT_MODEL, previous=None, canary=None, canary_weight=0.0)
    monkeypatch.setattr(api, "_load_slot", _fake_loader(api, base_model[0]))
    monkeypatch.setattr(api, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True); t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", api
    server.should_exit = True
    t.join(timeout=5)


class FakeMsg:
    def __init__(self, v):
        self.value = v


class FakeConsumer:
    """Stands in for KafkaConsumer: iterates over a fixed list of records."""
    def __init__(self, records):
        self.records = records

    def __call__(self, *a, **k):
        return self

    def __iter__(self):
        return (FakeMsg(r) for r in self.records)
