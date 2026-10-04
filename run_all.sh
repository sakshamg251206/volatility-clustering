#!/usr/bin/env bash
# Reproduce every table and figure. Raw data is cached in data/raw after the first download.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY -m pytest -q
for s in 01_download 02_stylized_facts 03_cross_asset 04_frequency 05_shocks 06_regimes 07_forecasting; do
  echo "== $s"; $PY scripts/$s.py > results/log_${s%%_*}.txt 2>&1
done
echo "done: results/tables, results/figures"
