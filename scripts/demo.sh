#!/usr/bin/env bash
# Fraudar demo control.
#   ./scripts/demo.sh up | down | restart | status | logs <name> | inject <A|B|C|U> | clear | scenario | wrongmodel
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
API=http://localhost:8000
RUN=.run
LOGS=logs/run
mkdir -p "$RUN" "$LOGS"

alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

start() {
  local name=$1; shift
  if alive "$name"; then echo "  $name already running"; return 0; fi
  setsid nohup "$@" > "$LOGS/$name.log" 2>&1 &
  echo $! > "$RUN/$name.pid"
  echo "  started $name"
}

stop() {
  if alive "$1"; then
    kill -TERM -- "-$(cat "$RUN/$1.pid")" 2>/dev/null || kill -TERM "$(cat "$RUN/$1.pid")" 2>/dev/null
    echo "  stopped $1"
  fi
  rm -f "$RUN/$1.pid"
}

wait_http() { for _ in $(seq 1 "$2"); do curl -sf "$1" > /dev/null 2>&1 && return 0; sleep 1; done; echo "  !! timeout waiting for $1"; return 1; }
wait_port() { for _ in $(seq 1 "$2"); do (echo > "/dev/tcp/localhost/$1") 2>/dev/null && return 0; sleep 1; done; echo "  !! port $1 not ready"; return 1; }
wait_log() { for _ in $(seq 1 "$3"); do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 1; done; echo "  !! timeout waiting for '$2' in $1"; return 1; }
active() { curl -sf "$API/health" | $PY -c "import sys,json; print(json.load(sys.stdin)['active'])" 2>/dev/null; }
wait_active() { for _ in $(seq 1 "$2"); do [ "$(active)" = "$1" ] && return 0; sleep 1; done; return 1; }

up() {
  echo "[1/5] Kafka, Prometheus, Grafana"
  docker compose up -d > /dev/null 2>&1 || { echo "  !! docker compose failed"; return 1; }
  wait_port 9092 90 || return 1

  echo "[2/5] MLflow"
  if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
    echo "  already running"
  else
    start mlflow .venv/bin/mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
    wait_http http://localhost:5000/health 90 || return 1
  fi

  echo "[3/5] API"
  if ! alive api && curl -sf "$API/health" > /dev/null 2>&1; then
    echo "  !! port 8000 is used by a process this script did not start; stop it (Ctrl+C) and retry"; return 1
  fi
  start api .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  wait_http "$API/health" 120 || { tail -n 15 "$LOGS/api.log"; return 1; }

  echo "[4/5] scoring consumer + orchestrator"
  start consumer $PY -u -m src.streaming.scoring_consumer
  start orchestrator $PY -u -m src.orchestrator.orchestrator
  wait_log "$LOGS/consumer.log" "Scoring consumer ready" 60 || return 1
  wait_log "$LOGS/orchestrator.log" "Orchestrator ready" 60 || return 1
  sleep 8   # let both consumer groups join before the first row is produced

  echo "[5/5] stream producer (warm-up takes about 100 s)"
  $PY -m src.streaming.inject --clear > /dev/null 2>&1
  start producer $PY -u -m src.streaming.drift_producer --tps 50 --warmup 5000 --rows 30000
  wait_log "$LOGS/orchestrator.log" "warm-up complete" 240 || return 1
  echo
  echo "READY. Dashboard: http://localhost:3000/d/fraudar-routing"
  echo "Next: ./scripts/demo.sh scenario   (or: inject A|B|C|U, wrongmodel)"
}

down() {
  for n in scenario producer orchestrator consumer api mlflow; do stop "$n"; done
  sleep 2
}

reset() {
  down
  local ts; ts=$(date +%Y%m%d-%H%M%S)
  mkdir -p "logs/archive/$ts"
  for f in logs/decisions.jsonl logs/predictions_audit.jsonl; do
    [ -f "$f" ] && mv "$f" "logs/archive/$ts/"
  done
  echo "  old logs archived to logs/archive/$ts"
}

status() {
  for n in mlflow api consumer orchestrator producer scenario; do
    if alive "$n"; then echo "  running  $n"; else echo "  stopped  $n"; fi
  done
  echo "  serving model: $(active)"
}

wrongmodel() {
  echo "wrong-model test: inject A, wait for specialist A, force spec-C (wrong), expect rollback to universal, end drift, expect v1"
  $PY -m src.streaming.inject --type A --frac 0.3 > /dev/null
  wait_active fraud-spec-A 90 || { echo "FAIL: router never switched to fraud-spec-A"; return 1; }
  echo "  router chose fraud-spec-A; settling 10 s"; sleep 10
  curl -sf -X POST "$API/admin/route" -H 'Content-Type: application/json' \
    -d '{"model":"fraud-spec-C","reason":"wrong-model test"}' > /dev/null
  echo "  forced fraud-spec-C (wrong model for drift A)"
  local t0; t0=$(date +%s)
  if wait_active fraud-universal 90; then
    echo "PASS: guardrail rolled back to fraud-universal after $(( $(date +%s) - t0 )) s"
  else
    echo "FAIL: no guardrail rollback within 90 s (serving $(active))"
  fi
  sleep 15
  $PY -m src.streaming.inject --clear > /dev/null
  if wait_active fraud-detection-model 120; then
    echo "PASS: policy returned to v1 after the drift ended"
  else
    echo "FAIL: still serving $(active) after the drift ended"
  fi
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  restart) reset; up ;;
  status) status ;;
  logs) tail -n 40 -f "$LOGS/${2:?usage: logs <mlflow|api|consumer|orchestrator|producer|scenario>}.log" ;;
  inject) $PY -m src.streaming.inject --type "${2:?usage: inject A|B|C|U}" --frac 0.3 ;;
  clear) $PY -m src.streaming.inject --clear ;;
  scenario)
    start scenario $PY -u -m src.streaming.inject --scenario demo --hold 45 --gap 40
    echo "  runs about 6 minutes; follow it with: ./scripts/demo.sh logs scenario" ;;
  wrongmodel) wrongmodel ;;
  *) sed -n '2,3p' "$0" ;;
esac
