"""Retraining + canary controller.

Watches the stream (its own Kafka group) and the API. When a drift model (a specialist or the
universal model) has been serving for --min-rows rows, it retrains on that fresh window:
  1. candidate = XGBoost on the newest drifted rows + clean rows from before the drift
  2. compare candidate vs the serving model on the newest 20% of the window (never trained on)
  3. if the candidate is at least as good: register as fraud-retrained@candidate, reload it in the
     API and send it --canary-weight of the traffic
  4. promote it to full traffic if its live F2 holds up against the serving model, otherwise drop it
Every decision goes to logs/retrain.jsonl and Prometheus (port 8003).
"""
import argparse
import json
import os
import time
from collections import deque
from datetime import datetime, timezone

import pandas as pd
import requests
from kafka import KafkaConsumer
from prometheus_client import Counter, Gauge, start_http_server

from src.retrain import trainer
from src.settings import API_URL, KAFKA, MLFLOW_URI, admin_headers

TOPIC = "transactions"
LOG = "logs/retrain.jsonl"
CANDIDATE = trainer.REGISTERED_NAME
DRIFT_MODELS = {"fraud-spec-A", "fraud-spec-B", "fraud-spec-C", "fraud-universal"}
PHASES = ["idle", "collecting", "training", "canary"]

RUNS = Counter("retrain_runs_total", "Retraining attempts by result", ["result"])
PHASE = Gauge("retrain_phase", "1 for the controller's current phase", ["phase"])
FRESH_ROWS = Gauge("retrain_fresh_rows", "Labeled rows collected since the current drift model was routed in")
CAND_F2 = Gauge("retrain_candidate_f2", "Candidate F2 on the held-out tail of the fresh window")
SERV_F2 = Gauge("retrain_serving_f2", "Serving model F2 on the same held-out tail")
CANARY_F2 = Gauge("retrain_canary_live_f2", "Live F2 of the canary (-1 = not enough rows)")


class Api:
    def __init__(self, url):
        self.url = url

    def get(self, path):
        r = requests.get(self.url + path, timeout=5)
        r.raise_for_status()
        return r.json()

    def post(self, path, body):
        r = requests.post(self.url + path, json=body, headers=admin_headers(), timeout=30)
        r.raise_for_status()
        return r.json()


def log(entry):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


class Controller:
    def __init__(self, api, args, load_serving=None, register=None):
        self.api, self.a = api, args
        self.load_serving = load_serving or (lambda name, alias: trainer.load_registered(name, alias, MLFLOW_URI))
        self.register = register or (lambda m, t, tags: trainer.register(m, t, tags, MLFLOW_URI))
        self.clean = deque(maxlen=args.clean_rows)
        self.fresh = deque(maxlen=args.max_rows)
        self.watching, self.phase, self.last_active = None, "idle", None
        self.canary_rows, self.last_retrain_row, self.rows = 0, -10**9, 0
        self._set_phase("idle")

    def _set_phase(self, p):
        self.phase = p
        for x in PHASES:
            PHASE.labels(phase=x).set(1 if x == p else 0)

    def on_record(self, rec):
        self.rows += 1
        if self.watching:
            self.fresh.append(rec)
            if self.phase == "canary":
                self.canary_rows += 1
        elif self.last_active in (None, "fraud-detection-model"):
            self.clean.append(rec)
        FRESH_ROWS.set(len(self.fresh))
        if self.rows % self.a.check_every == 0:
            self.tick()

    def tick(self):
        try:
            slots = self.api.get("/admin/slots")
        except Exception as e:
            print(f"  !! API unavailable: {e}", flush=True)
            return
        active = slots["active"]
        self.last_active = active
        if self.phase == "canary":
            return self._canary_step(slots)
        if active in DRIFT_MODELS:
            if active != self.watching:          # a new drift model was routed in: start a new window
                self.watching = active
                self.fresh.clear()
                self._set_phase("collecting")
            elif (len(self.fresh) >= self.a.min_rows
                  and self.rows - self.last_retrain_row >= self.a.cooldown_rows):
                self._retrain(slots)
        else:
            self.watching = None
            self.fresh.clear()
            self._set_phase("idle")

    def _retrain(self, slots):
        self._set_phase("training")
        self.last_retrain_row = self.rows
        serving = self.watching
        fresh = pd.DataFrame(list(self.fresh))
        clean = pd.DataFrame(list(self.clean)) if self.clean else pd.DataFrame()
        train_part, holdout = trainer.split(fresh, self.a.holdout)
        t0 = time.time()
        try:
            model, thr, feats = trainer.train_candidate(train_part, clean, n_estimators=self.a.trees)
            s_model, s_thr, s_feats = self.load_serving(serving, slots["slots"][serving]["alias"])
            s_thr = slots["slots"][serving]["threshold"]
            cand = trainer.f2_at(model, thr, holdout, feats)
            serv = trainer.f2_at(s_model, s_thr, holdout, s_feats)
        except Exception as e:
            RUNS.labels(result="failed").inc()
            log({"event": "retrain_failed", "serving": serving, "error": str(e)})
            self._set_phase("collecting")
            return
        CAND_F2.set(cand)
        SERV_F2.set(serv)
        base = {"serving": serving, "fresh_rows": len(fresh), "clean_rows": len(clean),
                "holdout_rows": len(holdout), "candidate_f2": round(cand, 4), "serving_f2": round(serv, 4),
                "threshold": round(thr, 4), "train_seconds": round(time.time() - t0, 1)}
        if cand < serv + self.a.min_gain:
            RUNS.labels(result="rejected_offline").inc()
            log({"event": "candidate_rejected", **base})
            self._set_phase("collecting")
            return
        try:
            version = self.register(model, thr, {"trained_while_serving": serving, "fresh_rows": len(fresh),
                                                 "holdout_f2": round(cand, 4)})
            self.api.post("/admin/reload", {"model": CANDIDATE})
            self.api.post("/admin/route", {"model": CANDIDATE, "weight": self.a.canary_weight,
                                           "reason": f"canary: retrained v{version} on fresh window"})
        except Exception as e:   # registry or API down: keep serving as is, try again after the cooldown
            RUNS.labels(result="failed").inc()
            log({"event": "deploy_failed", "error": str(e)[:300], **base})
            self._set_phase("collecting")
            return
        self.canary_rows = 0
        self._set_phase("canary")
        log({"event": "canary_started", "version": version, "weight": self.a.canary_weight, **base})

    def _canary_step(self, slots):
        if slots["canary"] != CANDIDATE or slots["active"] != self.watching:
            RUNS.labels(result="canary_interrupted").inc()
            log({"event": "canary_interrupted", "active": slots["active"]})
            self._end_canary()
            return
        perf = self.api.get("/admin/performance")["slots"]
        c, s = perf.get(CANDIDATE, {}), perf.get(self.watching, {})
        enough = c.get("n", 0) >= self.a.canary_min_rows and c.get("positives", 0) >= self.a.canary_min_pos
        CANARY_F2.set(c["f2"] if enough and c.get("f2") is not None else -1)
        if enough and c.get("f2") is not None and s.get("f2") is not None:
            if c["f2"] >= s["f2"] - self.a.tolerance:
                self.api.post("/admin/route", {"model": CANDIDATE, "weight": 1.0,
                                               "reason": f"canary promoted: live F2 {c['f2']:.2f} vs {s['f2']:.2f}"})
                RUNS.labels(result="promoted").inc()
                log({"event": "promoted", "canary_f2": round(c["f2"], 4), "serving_f2": round(s["f2"], 4),
                     "canary_rows": c["n"]})
            else:
                self.api.post("/admin/route", {"model": CANDIDATE, "weight": 0.0, "reason": "canary rejected"})
                RUNS.labels(result="rejected_live").inc()
                log({"event": "canary_rejected", "canary_f2": round(c["f2"], 4), "serving_f2": round(s["f2"], 4)})
            self._end_canary()
        elif self.canary_rows >= self.a.canary_max_rows:
            self.api.post("/admin/route", {"model": CANDIDATE, "weight": 0.0, "reason": "canary timed out"})
            RUNS.labels(result="canary_timeout").inc()
            log({"event": "canary_timeout", "canary_rows": c.get("n", 0)})
            self._end_canary()

    def _end_canary(self):
        CANARY_F2.set(-1)
        self.watching = None
        self.fresh.clear()
        self._set_phase("idle")


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=API_URL)
    ap.add_argument("--min-rows", type=int, default=6000, help="fresh rows before retraining (~2 min at 50 tx/s)")
    ap.add_argument("--max-rows", type=int, default=15000, help="newest rows kept for training")
    ap.add_argument("--clean-rows", type=int, default=5000, help="clean rows mixed in to keep normal behaviour")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--min-gain", type=float, default=0.0, help="candidate must beat serving F2 by this offline")
    ap.add_argument("--canary-weight", type=float, default=0.2)
    ap.add_argument("--canary-min-rows", type=int, default=150)
    ap.add_argument("--canary-min-pos", type=int, default=10)
    ap.add_argument("--canary-max-rows", type=int, default=12000)
    ap.add_argument("--tolerance", type=float, default=0.02, help="canary may trail serving F2 by this much")
    ap.add_argument("--cooldown-rows", type=int, default=15000)
    ap.add_argument("--check-every", type=int, default=250)
    ap.add_argument("--metrics-port", type=int, default=8003)
    ap.add_argument("--group", default=f"retrain-{int(time.time())}")
    return ap


def main(argv=None, consumer=None, **kw):
    a = parser().parse_args(argv)
    os.makedirs("logs", exist_ok=True)
    if a.metrics_port:
        start_http_server(a.metrics_port)
    for r in ("failed", "rejected_offline", "promoted", "rejected_live", "canary_timeout", "canary_interrupted"):
        RUNS.labels(result=r).inc(0)
    CANARY_F2.set(-1)
    ctl = Controller(Api(a.api), a, **kw)
    consumer = consumer or KafkaConsumer(
        TOPIC, bootstrap_servers=KAFKA, auto_offset_reset="latest", group_id=a.group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    print(f"Retrain controller ready (group {a.group}); retrains after {a.min_rows} rows on a drift model.",
          flush=True)
    for msg in consumer:
        ctl.on_record(msg.value)
    return ctl


if __name__ == "__main__":
    main()
