#!/usr/bin/env bash
# Download Prometheus + Alertmanager binaries (Linux/WSL, amd64) into .tools/ for the alert tests.
#   ./scripts/get_test_tools.sh && source .tools/env.sh && pytest
set -eu
cd "$(dirname "$0")/.."
mkdir -p .tools && cd .tools
P=3.5.0; A=0.28.1
curl -sL "https://github.com/prometheus/prometheus/releases/download/v$P/prometheus-$P.linux-amd64.tar.gz" | tar xz
curl -sL "https://github.com/prometheus/alertmanager/releases/download/v$A/alertmanager-$A.linux-amd64.tar.gz" | tar xz
cat > env.sh << ENV
export PROMTOOL=$PWD/prometheus-$P.linux-amd64/promtool
export PROMETHEUS_BIN=$PWD/prometheus-$P.linux-amd64/prometheus
export ALERTMANAGER_BIN=$PWD/alertmanager-$A.linux-amd64/alertmanager
export AMTOOL=$PWD/alertmanager-$A.linux-amd64/amtool
ENV
echo "done: source .tools/env.sh"
