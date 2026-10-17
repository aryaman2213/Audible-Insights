#!/bin/bash
# EC2 "User data" script for Amazon Linux 2023: paste it into Advanced details -> User data when launching the
# instance. On first boot it installs Docker, clones the repo, builds the image and serves the app on port 80.
# Progress log: /var/log/cloud-init-output.log
set -euxo pipefail

REPO_URL="https://github.com/aryaman2213/audible-insights.git"   # change if your repo name differs
APP_DIR=/opt/audible-insights

# 1 GB swap: a t2/t3.micro has 1 GB RAM, which is tight while pip builds the image.
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=1024
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

dnf install -y docker git
systemctl enable --now docker

git clone --depth 1 "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"
docker build -t audible-insights .
docker run -d --name audible-insights --restart unless-stopped -p 80:8501 audible-insights
