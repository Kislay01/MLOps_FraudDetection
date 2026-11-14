"""Alert rules and delivery.
- rule unit tests with promtool (tests/alerts_test.yml)
- the API webhook receiver
- live chain: real Prometheus + Alertmanager binaries -> API /alerts/webhook
Binary tests are skipped unless PROMETHEUS_BIN / ALERTMANAGER_BIN (or promtool on PATH) are set."""
import os
import shutil
import socket
import subprocess
import time

import pytest
import requests
import yaml

PROMTOOL = os.getenv("PROMTOOL") or shutil.which("promtool")
PROM = os.getenv("PROMETHEUS_BIN") or shutil.which("prometheus")
AM = os.getenv("ALERTMANAGER_BIN") or shutil.which("alertmanager")
AMTOOL = os.getenv("AMTOOL") or shutil.which("amtool")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.mark.skipif(not PROMTOOL, reason="promtool not installed")
def test_rule_unit_tests():
    for args in (["check", "rules", "prometheus/alerts.yml"], ["test", "rules", "tests/alerts_test.yml"]):
        r = subprocess.run([PROMTOOL, *args], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not AMTOOL, reason="amtool not installed")
def test_alertmanager_config():
    r = subprocess.run([AMTOOL, "check-config", "alertmanager/alertmanager.yml"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_webhook_receiver(api_client):
    c, api = api_client
    api.alerts.clear()
    body = {"alerts": [{"status": "firing", "labels": {"alertname": "FraudarGuardrailRollback", "severity": "critical"},
                        "annotations": {"summary": "Guardrail rolled back a failing model"}, "startsAt": "x"}]}
    assert c.post("/alerts/webhook", json=body).json()["stored"] == 1
    got = c.get("/alerts").json()["alerts"]
    assert got[0]["alertname"] == "FraudarGuardrailRollback" and got[0]["severity"] == "critical"
    for _ in range(120):
        c.post("/alerts/webhook", json=body)
    assert len(api.alerts) == 100                     # bounded


@pytest.mark.skipif(not (PROM and AM), reason="prometheus/alertmanager binaries not available")
def test_live_alert_chain(live_api, tmp_path):
    from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server
    url, api = live_api
    api.alerts.clear()
    reg = CollectorRegistry()
    guard = Counter("orch_guardrail_total", "x", ["model"], registry=reg)
    Gauge("orch_guardrail_f2", "x", registry=reg).set(-1)
    exp_port, prom_port, am_port, am_cluster = free_port(), free_port(), free_port(), free_port()
    start_http_server(exp_port, registry=reg)

    am_cfg = yaml.safe_load(open("alertmanager/alertmanager.yml"))
    am_cfg["route"].update(group_wait="1s", group_interval="2s")
    am_cfg["receivers"][0]["webhook_configs"][0]["url"] = f"{url}/alerts/webhook"
    (tmp_path / "am.yml").write_text(yaml.safe_dump(am_cfg))
    prom_cfg = {
        "global": {"scrape_interval": "1s", "evaluation_interval": "1s"},
        "rule_files": [os.path.abspath("prometheus/alerts.yml")],
        "alerting": {"alertmanagers": [{"static_configs": [{"targets": [f"127.0.0.1:{am_port}"]}]}]},
        "scrape_configs": [{"job_name": "orchestrator", "static_configs": [{"targets": [f"127.0.0.1:{exp_port}"]}]}],
    }
    (tmp_path / "prom.yml").write_text(yaml.safe_dump(prom_cfg))
    procs = [
        subprocess.Popen([AM, f"--config.file={tmp_path / 'am.yml'}", f"--storage.path={tmp_path / 'am'}",
                          f"--web.listen-address=127.0.0.1:{am_port}",
                          f"--cluster.listen-address=127.0.0.1:{am_cluster}"],
                         stdout=open(tmp_path / "am.log", "w"), stderr=subprocess.STDOUT),
        subprocess.Popen([PROM, f"--config.file={tmp_path / 'prom.yml'}", f"--storage.tsdb.path={tmp_path / 'tsdb'}",
                          f"--web.listen-address=127.0.0.1:{prom_port}"],
                         stdout=open(tmp_path / "prom.log", "w"), stderr=subprocess.STDOUT),
    ]
    try:
        guard.labels(model="fraud-spec-C").inc(0)       # pre-created at 0, as the orchestrator does
        for _ in range(60):                             # wait until Prometheus has scraped the 0
            try:
                q = requests.get(f"http://127.0.0.1:{prom_port}/api/v1/query",
                                 params={"query": "orch_guardrail_total"}, timeout=2).json()
                if q["data"]["result"]:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        time.sleep(2)
        guard.labels(model="fraud-spec-C").inc()          # the orchestrator rolls a model back
        deadline = time.time() + 60
        names = []
        while time.time() < deadline:
            names = [a["alertname"] for a in requests.get(f"{url}/alerts").json()["alerts"]]
            if "FraudarGuardrailRollback" in names:
                break
            time.sleep(1)
        if "FraudarGuardrailRollback" not in names:
            rules = requests.get(f"http://127.0.0.1:{prom_port}/api/v1/alerts").json()
            targets = requests.get(f"http://127.0.0.1:{prom_port}/api/v1/targets").json()
            pytest.fail(f"no alert delivered. prom alerts={rules} targets={targets} "
                        f"am.log={(tmp_path / 'am.log').read_text()[-1500:]}")
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)
