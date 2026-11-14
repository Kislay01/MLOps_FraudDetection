"""Admin endpoints: open when no token is configured (local demo), protected when it is."""


def test_predict_and_route_without_token(api_client, frame):
    c, api = api_client
    rec = frame.iloc[0].to_dict()
    r = c.post("/predict", json={"data": rec, "label": int(rec["isFraud"])})
    assert r.status_code == 200 and r.json()["served_by"] == "fraud-detection-model"
    r = c.post("/admin/route", json={"model": "fraud-spec-A", "reason": "test"})
    assert r.status_code == 200 and r.json()["active"] == "fraud-spec-A"
    assert c.post("/admin/rollback").json()["active"] == "fraud-detection-model"


def test_admin_requires_token_when_set(api_client, monkeypatch):
    c, api = api_client
    monkeypatch.setattr(api, "ADMIN_TOKEN", "s3cret")
    body = {"model": "fraud-spec-A", "reason": "test"}
    assert c.post("/admin/route", json=body).status_code == 401
    assert c.post("/admin/route", json=body, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert c.post("/admin/rollback").status_code == 401
    assert c.post("/admin/reload", json={"model": "fraud-spec-A"}).status_code == 401
    ok = c.post("/admin/route", json=body, headers={"X-Admin-Token": "s3cret"})
    assert ok.status_code == 200 and ok.json()["active"] == "fraud-spec-A"
    # read-only endpoints stay open for dashboards and the control room
    assert c.get("/admin/slots").status_code == 200
    assert c.get("/admin/performance").status_code == 200
    assert c.get("/health").status_code == 200


def test_orchestrator_sends_token(monkeypatch):
    import src.orchestrator.orchestrator as orch
    import src.settings as settings
    seen = {}

    class R:
        def raise_for_status(self):
            pass

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, headers=headers)
        return R()

    monkeypatch.setattr(settings, "ADMIN_TOKEN", "abc")
    monkeypatch.setattr(orch.requests, "post", fake_post)
    assert orch.set_route("http://api", "fraud-spec-B", "t")
    assert seen["headers"] == {"X-Admin-Token": "abc"}
