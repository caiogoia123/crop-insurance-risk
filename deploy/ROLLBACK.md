# Deploy and rollback (Oracle VM, Ubuntu 24.04 arm64)

Everything this project adds to the VM, and how to remove each piece. Run the
steps in order; after each one, check that the main site still answers 200:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://caiogoia.duckdns.org/
```

## What was added

| Piece | Where |
|---|---|
| Docker (Ubuntu package `docker.io`) | apt; service `docker` |
| Image `crop-risk:latest` | built from `/opt/crop-risk/app` (git clone of this repo) |
| Container `crop-risk-api` | `127.0.0.1:8001 -> 8000`, 1 CPU, 1.5 GB, rotated logs |
| Data, models, logs | `/opt/crop-risk/{data,models,logs,reports}` (owned by uid 10001) |
| Nginx rate-limit zone | `/etc/nginx/conf.d/crop-risk-ratelimit.conf` |
| Nginx location block | `/etc/nginx/sites-available/caiogoia`, between `# BEGIN crop-risk` and `# END crop-risk` (backup of the original next to it: `caiogoia.bak-croprisk`) |
| Retrain timer | `/etc/systemd/system/crop-risk-retrain.{service,timer}` |

## Roll back a bad model only

The previous bundles stay in `/opt/crop-risk/models/` (the last 3 are kept).

```bash
ls -t /opt/crop-risk/models/                     # pick the previous version
echo <version> | sudo tee /opt/crop-risk/models/CURRENT
sudo docker restart crop-risk-api
curl -s http://127.0.0.1:8001/health
```

## Remove everything

1. Stop the timer and remove it:
   ```bash
   sudo systemctl disable --now crop-risk-retrain.timer
   sudo rm /etc/systemd/system/crop-risk-retrain.service /etc/systemd/system/crop-risk-retrain.timer
   sudo systemctl daemon-reload
   ```
2. Remove the Nginx block and the rate-limit zone, test, reload:
   ```bash
   sudo cp /etc/nginx/sites-available/caiogoia /tmp/caiogoia.current
   sudo sed -i '/# BEGIN crop-risk/,/# END crop-risk/d' /etc/nginx/sites-available/caiogoia
   sudo rm /etc/nginx/conf.d/crop-risk-ratelimit.conf
   sudo nginx -t && sudo systemctl reload nginx
   ```
   If `nginx -t` fails, restore the original file: `sudo cp /etc/nginx/sites-available/caiogoia.bak-croprisk /etc/nginx/sites-available/caiogoia`
   (the backup was taken right before the block was added).
3. Stop and remove the container and images:
   ```bash
   sudo docker rm -f crop-risk-api crop-risk-retrain 2>/dev/null
   sudo docker image rm crop-risk:latest
   sudo docker image prune -f
   ```
4. Delete the project files:
   ```bash
   sudo rm -rf /opt/crop-risk
   ```
5. Uninstall Docker (optional):
   ```bash
   sudo apt-get purge -y docker.io && sudo apt-get autoremove -y
   sudo rm -rf /var/lib/docker /var/lib/containerd
   ```
   Docker adds its own iptables chains (DOCKER, DOCKER-USER, DOCKER-FORWARD) and sets
   the FORWARD policy to DROP; the INPUT rules for 22/80/443 are not touched. After
   purging, a reboot (or `sudo netfilter-persistent reload`) restores the saved rules
   without the Docker chains.

## Deploy (for reference)

```bash
# code and image
cd /opt/crop-risk/app && sudo git pull && sudo docker build -t crop-risk:latest .
# model bundle trained on the PC
scp -r models/<version> ubuntu@vm:/tmp/ && sudo mv /tmp/<version> /opt/crop-risk/models/
echo <version> | sudo tee /opt/crop-risk/models/CURRENT && sudo chown -R 10001:10001 /opt/crop-risk/models
# API (loopback only)
sudo /opt/crop-risk/app/deploy/run-api.sh
# Nginx: backup, insert block, nginx -t, reload (restores the backup if the test fails)
sudo /opt/crop-risk/app/deploy/nginx/install-location.sh
# timer
sudo cp deploy/systemd/crop-risk-retrain.* /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now crop-risk-retrain.timer
sudo docker image prune -f
```
