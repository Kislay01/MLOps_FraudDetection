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
