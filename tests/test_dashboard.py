"""Dashboard JSON: layout sanity, and every PromQL expression parses (needs promtool; skipped otherwise)."""
import json
import os
import shutil
import subprocess

import pytest
import yaml

DASH = "grafana/dashboards/fraudar_routing.json"
PROMTOOL = os.getenv("PROMTOOL") or shutil.which("promtool")


def load():
    return json.load(open(DASH))


def exprs():
    d = load()
    out = [t["expr"] for p in d["panels"] for t in p.get("targets", [])]
    out += [a["expr"] for a in d["annotations"]["list"]]
    return [e.replace("$__range", "5m") for e in out]


def test_layout():
    d = load()
    seen = {}
    for p in d["panels"]:
        g = p["gridPos"]
        assert g["x"] + g["w"] <= 24
        for x in range(g["x"], g["x"] + g["w"]):
            for y in range(g["y"], g["y"] + g["h"]):
                assert (x, y) not in seen, (p["title"], seen[(x, y)])
                seen[(x, y)] = p["title"]
    assert len({p["id"] for p in d["panels"]}) == len(d["panels"])


def test_serving_model_map_covers_retrained():
    d = load()
    p = next(p for p in d["panels"] if p["title"] == "Serving model")
    assert 'model="fraud-retrained"' in p["targets"][0]["expr"]
    texts = [v["text"] for m in p["fieldConfig"]["defaults"]["mappings"] for v in m["options"].values()]
    assert "Retrained model" in texts


@pytest.mark.skipif(not PROMTOOL, reason="promtool not installed")
def test_promql_parses(tmp_path):
    rules = {"groups": [{"name": "dash", "rules": [{"record": f"dash:expr{i}", "expr": e}
                                                    for i, e in enumerate(exprs())]}]}
    f = tmp_path / "r.yml"
    f.write_text(yaml.safe_dump(rules))
    r = subprocess.run([PROMTOOL, "check", "rules", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
