#!/bin/bash
# Redeploy the latest code on the EC2 instance:  sudo bash /opt/audible-insights/deploy/update.sh
set -euxo pipefail
cd /opt/audible-insights
git pull --ff-only
docker build -t audible-insights .
docker rm -f audible-insights || true
docker run -d --name audible-insights --restart unless-stopped -p 80:8501 audible-insights
docker image prune -f
