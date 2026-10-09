#!/usr/bin/env bash
# Add the /crop-risk/ block to the site's vhost (idempotent, self-reverting).
# Backs up the vhost, inserts the block right after "location / { ... }" in the
# HTTPS server, runs `nginx -t`, and restores the backup if the test fails.
set -euo pipefail
VHOST=/etc/nginx/sites-available/caiogoia
HERE=$(cd "$(dirname "$0")" && pwd)
BLOCK=$HERE/crop-risk-location.conf
ZONE=/etc/nginx/conf.d/crop-risk-ratelimit.conf

if grep -q "# BEGIN crop-risk" "$VHOST"; then
  echo "crop-risk block already present"
  exit 0
fi
cp "$VHOST" "$VHOST.bak-croprisk"
cp "$HERE/crop-risk-ratelimit.conf" "$ZONE"
awk -v blk="$BLOCK" '
  { print }
  /location \/ \{ try_files/ && !done { while ((getline line < blk) > 0) print line; done = 1 }
' "$VHOST.bak-croprisk" > "$VHOST.new"
grep -q "# BEGIN crop-risk" "$VHOST.new" || { echo "anchor not found, nothing changed"; rm -f "$VHOST.new" "$ZONE"; exit 1; }
mv "$VHOST.new" "$VHOST"

if nginx -t; then
  systemctl reload nginx
  echo "installed and reloaded"
else
  echo "nginx -t failed: restoring the original vhost"
  cp "$VHOST.bak-croprisk" "$VHOST"
  rm -f "$ZONE"
  nginx -t && systemctl reload nginx
  exit 1
fi
