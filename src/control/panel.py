"""Fraudar control room: buttons that drive scripts/demo.sh, plus live status.
Start with: ./scripts/demo.sh panel   then open http://localhost:8100"""
import json
import os
import subprocess
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[2]
RUN, LOGS = ROOT / ".run", ROOT / "logs" / "run"
API, MLFLOW = "http://localhost:8000", "http://localhost:5000"
ACTIONS = {
    "up": ["up"], "restart": ["restart"], "down": ["down"],
    "scenario": ["scenario"], "wrongmodel": ["wrongmodel"], "clear": ["clear"],
    "inject-A": ["inject", "A"], "inject-B": ["inject", "B"],
    "inject-C": ["inject", "C"], "inject-U": ["inject", "U"],
}
state = {"name": None, "proc": None, "started": 0.0}
app = FastAPI(title="Fraudar control room")


def _pid_alive(name):
    try:
        os.kill(int((RUN / f"{name}.pid").read_text()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def _get(url, timeout=1.5):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json() if "json" in r.headers.get("content-type", "") else {"ok": True}
    except Exception:
        return None


def _last_decision():
    f = ROOT / "logs" / "decisions.jsonl"
    try:
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            last = fh.read().decode(errors="ignore").strip().splitlines()[-1]
        d = json.loads(last)
        d["age"] = round(time.time() - f.stat().st_mtime)
        return d
    except Exception:
        return None


@app.get("/api/status")
def status():
    proc = state["proc"]
    running = proc is not None and proc.poll() is None
    slots = _get(f"{API}/admin/slots")
    perf = _get(f"{API}/admin/performance")
    active = slots["active"] if slots else None
    log = LOGS / "action.log"
    return {
        "services": {"mlflow": _get(f"{MLFLOW}/health") is not None, "api": slots is not None,
                     **{n: _pid_alive(n) for n in ("consumer", "orchestrator", "producer", "scenario")}},
        "active": active,
        "history": (slots or {}).get("history", [])[-8:],
        "perf": ((perf or {}).get("slots") or {}).get(active),
        "decision": _last_decision(),
        "action": {"name": state["name"], "running": running,
                   "seconds": round(time.time() - state["started"]) if running else None},
        "log": log.read_text(errors="ignore").splitlines()[-30:] if log.exists() else [],
    }


@app.post("/api/action/{name}")
def action(name: str):
    if name not in ACTIONS:
        raise HTTPException(status_code=404, detail="unknown action")
    proc = state["proc"]
    if proc is not None and proc.poll() is None:
        raise HTTPException(status_code=409, detail=f"'{state['name']}' is still running")
    LOGS.mkdir(parents=True, exist_ok=True)
    out = open(LOGS / "action.log", "w")
    out.write(f"$ demo.sh {' '.join(ACTIONS[name])}\n")
    out.flush()
    state.update(name=name, started=time.time(),
                 proc=subprocess.Popen(["bash", str(ROOT / "scripts" / "demo.sh"), *ACTIONS[name]],
                                       cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                       start_new_session=True))
    return {"started": name}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraudar Control Room</title>
<style>
  :root{--bg:#0d0d0d;--panel:#1a1a19;--line:#2c2c2a;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
        --blue:#3987e5;--orange:#d95926;--aqua:#199e70;--yellow:#c98500;--magenta:#d55181;
        --good:#0ca30c;--warn:#fab219;--crit:#d03b3b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
  header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;
         padding:14px 20px;border-bottom:1px solid var(--line)}
  h1{font-size:18px;margin:0;font-weight:650}
  header a{color:var(--blue);font-weight:600;text-decoration:none}
  main{max-width:1120px;margin:0 auto;padding:16px;display:grid;gap:16px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
  .card h2{margin:0 0 10px;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:650}
  .tile{border-radius:8px;padding:16px 10px;text-align:center;font-weight:700;font-size:18px;background:#383835;color:#fff}
  .sub{margin-top:8px;font-size:12px;color:var(--ink2);min-height:18px}
  .row{display:flex;flex-wrap:wrap;gap:8px}
  button{font:inherit;font-weight:650;color:#fff;background:#2c2c2a;border:1px solid #383835;border-radius:8px;padding:10px 14px;cursor:pointer}
  button:hover:not(:disabled){filter:brightness(1.25)}
  button:disabled{opacity:.4;cursor:not-allowed}
  button.A{background:var(--orange);border-color:var(--orange)}
  button.B{background:var(--aqua);border-color:var(--aqua)}
  button.C{background:var(--yellow);border-color:var(--yellow);color:#0d0d0d}
  button.U{background:var(--magenta);border-color:var(--magenta)}
  button.go{background:var(--blue);border-color:var(--blue)}
  button.stop{background:transparent;border-color:var(--crit);color:#ff8a8a}
  .group{display:grid;gap:8px;align-content:start}
  .group h3{margin:0;font-size:12px;color:var(--muted);font-weight:650}
  .dots{display:flex;flex-wrap:wrap;gap:6px 14px}
  .dot{display:inline-flex;align-items:center;gap:6px;color:var(--ink2);font-size:13px}
  .dot i{width:9px;height:9px;border-radius:50%;background:#555;display:inline-block}
  .dot.on i{background:var(--good)}
  ul{list-style:none;margin:0;padding:0;display:grid;gap:6px}
  li{display:flex;gap:10px;align-items:baseline;font-size:13px;color:var(--ink2)}
  li time{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
  li b{color:var(--ink)}
  pre{margin:0;max-height:240px;overflow:auto;background:#0d0d0d;border:1px solid var(--line);border-radius:8px;
      padding:10px;color:var(--ink2);font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap}
  #busy{margin-top:12px;color:var(--warn);font-weight:600;min-height:20px}
</style>
</head>
<body>
<header>
  <h1>Fraudar · Control room</h1>
  <a href="http://localhost:3000/d/fraudar-routing" target="_blank" rel="noopener">Open live dashboard ↗</a>
</header>
<main>
  <section class="grid">
    <div class="card"><h2>Serving model</h2><div class="tile" id="serving">…</div><div class="sub" id="servingSub"></div></div>
    <div class="card"><h2>Detected drift</h2><div class="tile" id="detected">…</div><div class="sub" id="detectedSub"></div></div>
    <div class="card"><h2>Orchestrator</h2><div class="tile" id="orch">…</div><div class="sub" id="orchSub"></div></div>
    <div class="card"><h2>Guardrail · live F2</h2><div class="tile" id="guard">…</div><div class="sub" id="guardSub"></div></div>
  </section>

  <section class="card">
    <h2>Controls</h2>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(280px,1fr))">
      <div class="group"><h3>Stack</h3><div class="row">
        <button class="go" data-act="up">Start demo</button>
        <button data-act="restart">Reset &amp; restart</button>
        <button class="stop" data-act="down">Stop all</button>
      </div></div>
      <div class="group"><h3>Show the system</h3><div class="row">
        <button class="go" data-act="scenario">Run full scenario (~6 min)</button>
        <button data-act="wrongmodel">Wrong-model test</button>
      </div></div>
      <div class="group"><h3>Inject a drift</h3><div class="row">
        <button class="A" data-act="inject-A">A · amount</button>
        <button class="B" data-act="inject-B">B · device</button>
        <button class="C" data-act="inject-C">C · velocity</button>
        <button class="U" data-act="inject-U">U · universal</button>
        <button data-act="clear">Clear drift</button>
      </div></div>
    </div>
    <div id="busy"></div>
  </section>

  <section class="grid" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))">
    <div class="card"><h2>Services</h2><div class="dots" id="services"></div>
      <h2 style="margin-top:16px">Routing events</h2><ul id="events"></ul></div>
    <div class="card"><h2>Last action output</h2><pre id="log"></pre></div>
  </section>
</main>
<script>
const MODEL = {
  "fraud-detection-model": ["v1 · general model", "#3987e5"],
  "fraud-spec-A": ["Specialist A · amount", "#d95926"],
  "fraud-spec-B": ["Specialist B · device/email", "#199e70"],
  "fraud-spec-C": ["Specialist C · velocity", "#c98500"],
  "fraud-universal": ["Universal model", "#d55181"],
};
const DRIFT = {
  normal: ["Normal traffic", "#3987e5"],
  A: ["A · amount (card testing)", "#d95926"],
  B: ["B · device/email (takeover)", "#199e70"],
  C: ["C · velocity (fraud ring)", "#c98500"],
  U: ["U · universal shift", "#d55181"],
};
const GOOD = "#1f8f1f", CRIT = "#d03b3b", MUTED = "#383835";
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
function tile(id, text, color) { const e = $(id); e.textContent = text; e.style.background = color || MUTED; }
const clock = ts => new Date(ts).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
let errMsg = "";

async function refresh() {
  let s;
  try { s = await (await fetch("/api/status")).json(); }
  catch (e) { tile("serving", "Control room offline"); return; }

  const m = MODEL[s.active];
  tile("serving", m ? m[0] : "API offline", m ? m[1] : MUTED);
  $("servingSub").textContent = s.active ? "Answering /predict right now" : "Press Start demo to bring the stack up";

  const d = s.decision, live = d && d.age < 30;
  if (live) {
    const t = DRIFT[d.label] || DRIFT.normal;
    tile("detected", t[0], t[1]);
    $("detectedSub").textContent = `A ${d.scores.A.toFixed(2)} · B ${d.scores.B.toFixed(2)} · C ${d.scores.C.toFixed(2)} · breadth ${d.breadth}`;
    if (d.state === "normal") tile("orch", "MONITORING", GOOD);
    else tile("orch", "ROUTED · " + (DRIFT[d.state] || ["drift"])[0].split(" ")[0], (DRIFT[d.state] || DRIFT.normal)[1]);
    $("orchSub").textContent = `${d.rows_seen.toLocaleString()} rows seen`;
  } else {
    tile("detected", "No monitoring data"); $("detectedSub").textContent = "";
    tile("orch", "IDLE"); $("orchSub").textContent = "";
  }

  const p = s.perf;
  if (p && p.n >= 150 && p.positives >= 10 && p.f2 !== null) {
    tile("guard", p.f2.toFixed(2), p.f2 >= 0.5 ? GOOD : CRIT);
    const pos = p.tp + p.fn;
    $("guardSub").textContent = `last ${p.n} rows · ${p.positives} fraud · recall ${(pos ? p.tp / pos : 0).toFixed(2)}`;
  } else {
    tile("guard", "Collecting evidence");
    $("guardSub").textContent = p ? `${p.n} rows since routed in` : "";
  }

  const names = {mlflow: "MLflow", api: "API", consumer: "Scorer", orchestrator: "Orchestrator", producer: "Stream", scenario: "Scenario"};
  $("services").innerHTML = Object.entries(names)
    .map(([k, v]) => `<span class="dot ${s.services[k] ? "on" : ""}"><i></i>${v}</span>`).join("");

  $("events").innerHTML = s.history.slice().reverse().map(h => {
    const mm = MODEL[h.model];
    return `<li><time>${clock(h.timestamp)}</time><span><b>${esc(mm ? mm[0] : h.model)}</b> · ${esc(h.event)} · ${esc(h.reason)}</span></li>`;
  }).join("") || "<li>No routing events yet</li>";

  const a = s.action;
  document.querySelectorAll("button[data-act]").forEach(b => b.disabled = a.running);
  const btn = a.name && document.querySelector(`button[data-act="${a.name}"]`);
  $("busy").textContent = a.running ? `Running: ${btn ? btn.textContent : a.name} · ${a.seconds}s` : errMsg;

  const lg = $("log");
  lg.textContent = s.log.join("\n");
  lg.scrollTop = lg.scrollHeight;
}

async function act(name) {
  errMsg = "";
  const r = await fetch("/api/action/" + name, {method: "POST"});
  if (!r.ok) { const e = await r.json().catch(() => ({})); errMsg = e.detail || "Action failed"; }
  refresh();
}
document.querySelectorAll("button[data-act]").forEach(b => b.addEventListener("click", () => act(b.dataset.act)));
setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
