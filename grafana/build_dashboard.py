"""Generate the Fraudar Grafana dashboard as provisioned JSON."""
import json
import pathlib

DS = {"type": "prometheus", "uid": "fraudar-prom"}

# Colour follows the entity everywhere (dark-surface categorical slots, fixed order).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
MUTED, INK, WHITE = "#898781", "#c3c2b7", "#ffffff"

DRIFT_MAP = {
    0: ("Normal traffic", BLUE), 1: ("A · amount (card testing)", ORANGE),
    2: ("B · device/email (takeover)", AQUA), 3: ("C · velocity (fraud ring)", YELLOW),
    4: ("U · universal shift", MAGENTA),
}
MODEL_MAP = {
    0: ("v1 · general model", BLUE), 1: ("Specialist A · amount", ORANGE),
    2: ("Specialist B · device/email", AQUA), 3: ("Specialist C · velocity", YELLOW),
    4: ("Universal model", MAGENTA),
}
MODELS = ["fraud-detection-model", "fraud-spec-A", "fraud-spec-B", "fraud-spec-C", "fraud-universal"]
MODEL_COLOR = dict(zip(MODELS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))

_id = 0


def nid():
    global _id
    _id += 1
    return _id


def tgt(expr, legend="", ref="A", instant=False):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref,
            "instant": instant, "range": not instant}


def fixed(color):
    return {"mode": "fixed", "fixedColor": color}


def by_name(name, props):
    return {"matcher": {"id": "byName", "options": name}, "properties": props}


def color_ov(name, color, width=None):
    props = [{"id": "color", "value": fixed(color)}]
    if width:
        props.append({"id": "custom.lineWidth", "value": width})
    return by_name(name, props)


def dashed_ov(name, color=MUTED):
    return by_name(name, [
        {"id": "color", "value": fixed(color)},
        {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 8]}},
        {"id": "custom.lineWidth", "value": 1},
    ])


def value_map(m):
    return [{"type": "value", "options": {
        str(k): {"text": t, "color": c, "index": k} for k, (t, c) in m.items()}}]


def code_expr(metric, label, keys):
    parts = []
    for i, k in enumerate(keys, start=1):
        s = f'sum({metric}{{{label}="{k}"}})'
        parts.append(s if i == 1 else f"{i}*{s}")
    return " + ".join(parts)


def panel(ptype, title, x, y, w, h, targets, desc, options=None, defaults=None, overrides=None):
    return {"id": nid(), "type": ptype, "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
            "options": options or {}, "fieldConfig": {"defaults": defaults or {}, "overrides": overrides or []}}


STAT_OPTS = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
             "graphMode": "none", "textMode": "value", "justifyMode": "center",
             "orientation": "auto", "text": {"valueSize": 20}}


def stat_map(title, x, w, expr, mapping, desc):
    return panel("stat", title, x, 0, w, 4, [tgt(expr, instant=True)], desc,
                 {**STAT_OPTS, "colorMode": "background"},
                 {"mappings": value_map(mapping), "color": fixed(MUTED)})


def stat_plain(title, x, y, w, h, expr, desc, unit="none", instant=True, graph="none", color=INK, decimals=None):
    d = {"unit": unit, "color": fixed(color)}
    if decimals is not None:
        d["decimals"] = decimals
    return panel("stat", title, x, y, w, h, [tgt(expr, instant=instant)], desc,
                 {**STAT_OPTS, "colorMode": "value", "graphMode": graph}, d)


def ts(title, x, y, w, h, targets, desc, unit=None, overrides=None, stack=False, minv=None, maxv=None, fill=0):
    custom = {"lineWidth": 2, "fillOpacity": fill, "showPoints": "never", "spanNulls": False,
              "gradientMode": "none", "axisBorderShow": False}
    if stack:
        custom["stacking"] = {"mode": "normal", "group": "A"}
    d = {"custom": custom, "color": {"mode": "palette-classic"}}
    if unit:
        d["unit"] = unit
    if minv is not None:
        d["min"] = minv
    if maxv is not None:
        d["max"] = maxv
    opts = {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"}}
    return panel("timeseries", title, x, y, w, h, targets, desc, opts, d, overrides)


def range_map(m):
    return [{"type": "range", "options": {
        "from": k - 0.5, "to": k + 0.5, "result": {"text": t, "color": c, "index": k}}}
        for k, (t, c) in m.items()]

def timeline(title, x, y, w, h, targets, mapping, desc):
    opts = {"mergeValues": True, "showValue": "auto", "alignValue": "center", "rowHeight": 0.85,
            "legend": {"showLegend": False}, "tooltip": {"mode": "single"}}
    d = {"custom": {"fillOpacity": 85, "lineWidth": 0}, "color": fixed(MUTED),
         "mappings": range_map(mapping)}
    return panel("state-timeline", title, x, y, w, h, targets, desc, opts, d)

def oc(outcome, by=False, win="45s"):
    g = " by (served_by)" if by else ""
    return f'sum{g}(increase(fraud_labeled_outcomes_total{{outcome="{outcome}"}}[{win}]))'


def f2_expr(by=False):
    tp, fn, fp = oc("tp", by), oc("fn", by), oc("fp", by)
    if by:
        return f"5*{tp} / (5*{tp} + 4*{fn} + {fp})"
    z = lambda e: f"({e} or vector(0))"
    return f"5*{z(tp)} / (5*{z(tp)} + 4*{z(fn)} + {z(fp)})"


def prec_expr():
    z = lambda e: f"({e} or vector(0))"
    return f"{z(oc('tp'))} / ({z(oc('tp'))} + {z(oc('fp'))})"


def rec_expr():
    z = lambda e: f"({e} or vector(0))"
    return f"{z(oc('tp'))} / ({z(oc('tp'))} + {z(oc('fn'))})"


model_expr = code_expr("fraud_active_model", "model", MODELS[1:])
injected_expr = code_expr("stream_injected_mode", "mode", ["A", "B", "C", "U"])
detected_expr = code_expr("orch_detected_label", "label", ["A", "B", "C", "U"])

f2_defaults_panel = panel(
    "stat", "Live F2 score", 16, 0, 4, 4, [tgt(f2_expr(False), instant=True)],
    "F2 over the last 45 s of labeled predictions. F2 weights recall higher than precision, "
    "which is what fraud teams care about. Red < 0.3, amber < 0.5, green >= 0.5.",
    {**STAT_OPTS, "colorMode": "background"},
    {"unit": "none", "decimals": 2, "min": 0, "max": 1, "color": {"mode": "thresholds"},
     "thresholds": {"mode": "absolute", "steps": [
         {"color": CRIT, "value": None}, {"color": WARN, "value": 0.3}, {"color": GOOD, "value": 0.5}]}})

panels = [
    stat_map("Serving model", 0, 5, model_expr, MODEL_MAP,
             "The model currently answering /predict. Changes automatically when the orchestrator reroutes."),
    stat_map("Detected drift", 5, 4, detected_expr, DRIFT_MAP,
             "What the orchestrator's fingerprint says about the latest window of 1,000 transactions."),
    stat_map("Injected drift (truth)", 9, 4, injected_expr, DRIFT_MAP,
             "Ground truth from the stream producer. The system never reads this; it is shown only to compare."),
    panel("stat", "Orchestrator", 13, 0, 3, 4, [tgt('sum(orch_policy_state{state="warming_up"})', instant=True)],
          "Warm-up: the first 5,000 transactions become the clean reference before monitoring begins.",
          {**STAT_OPTS, "colorMode": "background"},
          {"mappings": value_map({0: ("MONITORING", GOOD), 1: ("WARMING UP", WARN)}), "color": fixed(MUTED)}),
    f2_defaults_panel,
    stat_plain("Throughput", 20, 0, 4, 4, "sum(rate(fraud_predictions_total[30s]))",
               "Predictions per second served by the API.", unit="reqps", instant=False, graph="area", decimals=1),

    ts("1 · Drift signal per feature group", 0, 4, 14, 9,
       [tgt("orch_drift_score", "Group {{group}}", "A"), tgt("orch_group_threshold", "Alarm threshold", "B")],
       "Each line is how far a feature group has moved from recent normal traffic, in units of its noise "
       "floor. A = transaction amount, B = device/email/product, C = velocity. Crossing the dashed line "
       "means that group is drifting.",
       overrides=[color_ov("Group A", ORANGE), color_ov("Group B", AQUA), color_ov("Group C", YELLOW),
                  dashed_ov("Alarm threshold")], minv=0),
    ts("2 · Breadth: features drifting at once", 14, 4, 10, 9,
       [tgt("orch_drift_breadth", "Features drifting at once", "A"),
        tgt("orch_breadth_threshold", "Alarm threshold", "B")],
       "A broad shift touches dozens of features at once. Crossing the dashed line classifies the drift as "
       "universal (U) instead of a single specialist type.",
       overrides=[color_ov("Features drifting at once", MAGENTA), dashed_ov("Alarm threshold")], minv=0),

    timeline("3 · Which model is serving", 0, 13, 12, 6, [tgt(model_expr, "Serving model")], MODEL_MAP,
             "The active route over time. Every change here was made by the orchestrator, not a person."),
    timeline("4 · Injected vs. detected drift", 12, 13, 12, 6,
             [tgt(injected_expr, "Injected (truth)", "A"), tgt(detected_expr, "Detected (fingerprint)", "B")],
             DRIFT_MAP,
             "Top lane: what was injected. Bottom lane: what the system detected. The gap is detection delay."),

    ts("5 · Live F2 score", 0, 19, 12, 9,
       [tgt(f2_expr(False), "Overall", "A"), tgt(f2_expr(True), "{{served_by}}", "B")],
       "F2 over a 45 s window. The white line is the overall score; colored lines are per model. "
       "Watch it dip when drift starts and recover after the switch.",
       overrides=[color_ov("Overall", WHITE, 3)] + [color_ov(m, c) for m, c in MODEL_COLOR.items()],
       minv=0, maxv=1),
    ts("6 · Precision and recall", 12, 19, 12, 9,
       [tgt(prec_expr(), "Precision", "A"), tgt(rec_expr(), "Recall", "B")],
       "Precision: share of flagged transactions that are fraud. Recall: share of fraud that gets caught.",
       overrides=[color_ov("Precision", WHITE), color_ov("Recall", MUTED)], minv=0, maxv=1),

    ts("7 · Traffic share by model", 0, 28, 8, 7,
       [tgt("sum by (served_by)(rate(fraud_predictions_total[30s]))", "{{served_by}}")],
       "Requests per second answered by each model.", unit="reqps", stack=True, fill=60,
       overrides=[color_ov(m, c) for m, c in MODEL_COLOR.items()]),
    ts("8 · Inference latency (p95)", 8, 28, 8, 7,
       [tgt("histogram_quantile(0.95, sum by (le)(rate(fraud_prediction_latency_seconds_bucket[1m]))) * 1000",
            "p95 latency")],
       "95th-percentile time to score one transaction inside the API.", unit="ms",
       overrides=[color_ov("p95 latency", INK)], minv=0),
    stat_plain("Route switches", 16, 28, 4, 7, "sum(increase(fraud_route_switches_total[$__range]))",
               "Automatic model switches in the selected time range.", decimals=0),
    stat_plain("Orchestrator API errors", 20, 28, 4, 7, "sum(orch_api_errors_total) or vector(0)",
               "Failed routing calls from the orchestrator to the API. Should stay at 0.", decimals=0),
]

panels += [
    ts("9 · Fraud flag rate", 0, 35, 8, 7,
       [tgt('sum(rate(fraud_predictions_total{predicted_class="1"}[1m])) / sum(rate(fraud_predictions_total[1m]))',
            "Flagged as fraud")],
       "Share of transactions the serving model flags as fraud. It rises during a drift because the "
       "injected fraud raises the true fraud rate.",
       unit="percentunit", overrides=[color_ov("Flagged as fraud", WHITE)], minv=0),
    ts("10 · Predictions by class", 8, 35, 8, 7,
       [tgt("sum by (predicted_class)(rate(fraud_predictions_total[30s]))", "class {{predicted_class}}")],
       "Requests per second predicted legitimate (class 0) versus fraud (class 1).",
       unit="reqps", stack=True, fill=60,
       overrides=[color_ov("class 0", MUTED), color_ov("class 1", WHITE)]),
    panel("heatmap", "11 · Prediction confidence distribution", 16, 35, 8, 7,
          [{**tgt("sum by (le)(increase(fraud_prediction_confidence_bucket{le!='+Inf'}[1m]))", "{{le}}"), "format": "heatmap"}],
          "How sure the serving model is about each transaction, from 0 (legitimate) to 1 (fraud). "
          "Brighter cells mean more transactions at that confidence.",
          {"calculate": False, "cellGap": 1, "yAxis": {"axisPlacement": "left"},
           "color": {"mode": "scheme", "scheme": "Blues", "fill": "dark-blue", "scale": "exponential",
                     "exponent": 0.5, "steps": 64, "reverse": False},
           "legend": {"show": False}, "tooltip": {"mode": "single", "yHistogram": False},
           "rowsFrame": {"layout": "auto"}},
          {}),
]

dashboard = {
    "uid": "fraudar-routing",
    "title": "Fraudar · Live Drift Routing",
    "tags": ["fraudar"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "editable": True,
    "refresh": "5s",
    "time": {"from": "now-10m", "to": "now"},
    "templating": {"list": []},
    "annotations": {"list": [{
        "datasource": DS, "enable": True, "name": "Route switches", "iconColor": WHITE,
        "expr": "increase(fraud_route_switches_total[10s]) > 0", "step": "5s",
        "titleFormat": "Route switch", "textFormat": "{{from_model}} → {{to_model}}",
    }]},
    "panels": panels,
}

out = pathlib.Path("grafana/dashboards/fraudar_routing.json")
out.write_text(json.dumps(dashboard, indent=2))
print(f"wrote {out} ({len(panels)} panels)")