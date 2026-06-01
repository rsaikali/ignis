#!/usr/bin/env bash
# Nightly on-device retrain wrapper for cron (Raspberry Pi).
#
# Install (on the Pi):
#   crontab -e
#   # 03:17 every night -- off-peak; odd minute to avoid round-hour contention
#   17 3 * * *  /home/rsaikali/ignis/scripts/cron-retrain.sh >> /home/rsaikali/ignis/retrain.log 2>&1
#
# A flock guard prevents overlap if a pass runs long.
set -euo pipefail

IGNIS_DIR="${IGNIS_DIR:-/home/rsaikali/ignis}"
LOCK="/tmp/ignis-retrain.lock"

cd "$IGNIS_DIR"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date -Is) retrain: previous run still active, skipping"
    exit 0
fi

echo "$(date -Is) retrain: start"
docker compose --profile train run --rm train
echo "$(date -Is) retrain: done"
