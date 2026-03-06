#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/ufc-elf}"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx git

sudo mkdir -p "$APP_DIR"
sudo chown -R "$USER":"$USER" "$APP_DIR"

cd "$APP_DIR"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  cp .env.example .env
fi

sudo cp deploy/systemd/ufc-elf.service /etc/systemd/system/ufc-elf.service
sudo systemctl daemon-reload
sudo systemctl enable ufc-elf
sudo systemctl restart ufc-elf

sudo cp deploy/nginx/ufc-elf.conf /etc/nginx/sites-available/ufc-elf
sudo ln -sf /etc/nginx/sites-available/ufc-elf /etc/nginx/sites-enabled/ufc-elf
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo "Deployment bootstrap complete."
