#!/usr/bin/env bash
# Monthly retrain on the VM. Runs the challenger in a throw-away container with
# low priority and capped resources. The champion stays in place unless the
# challenger is promoted AND the restarted API passes its health check.
set -uo pipefail
BASE=/opt/crop-risk
LOG=$BASE/logs/retrain.log
IMAGE=crop-risk:latest
mkdir -p "$BASE/logs"
exec >>"$LOG" 2>&1
echo "=== $(date -Is) retrain start"
prev=$(cat "$BASE/models/CURRENT")

docker run --rm --name crop-risk-retrain --network host \
  --memory 6g --memory-swap 6g --cpus 1.5 \
  --log-driver json-file --log-opt max-size=10m --log-opt max-file=2 \
  -v "$BASE/data:/app/data" -v "$BASE/models:/app/models" \
  -v "$BASE/logs:/app/logs" -v "$BASE/reports:/app/reports" \
  "$IMAGE" nice -n 19 ionice -c 3 python -m croprisk.retrain
rc=$?

if [ "$rc" -eq 10 ]; then
  echo "promoted $(cat "$BASE/models/CURRENT"); restarting API"
  docker restart crop-risk-api
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8001/health; then echo; echo "health ok"; exit 0; fi
    sleep 2
  done
  echo "health check failed: rolling back to $prev"
  echo "$prev" > "$BASE/models/CURRENT"
  docker restart crop-risk-api
  exit 1
elif [ "$rc" -ne 0 ]; then
  echo "retrain failed (rc=$rc); champion $prev kept"
  exit "$rc"
fi
echo "champion $prev kept"
