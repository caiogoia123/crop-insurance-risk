#!/usr/bin/env bash
# (Re)create the API container: loopback only, capped resources, rotated logs.
set -euo pipefail
BASE=/opt/crop-risk
docker rm -f crop-risk-api >/dev/null 2>&1 || true
docker run -d --name crop-risk-api --restart unless-stopped \
  --memory 1.5g --memory-swap 1.5g --cpus 1.0 \
  --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
  -p 127.0.0.1:8001:8000 \
  -e ROOT_PATH=/crop-risk \
  -v "$BASE/models:/app/models:ro" \
  crop-risk:latest
