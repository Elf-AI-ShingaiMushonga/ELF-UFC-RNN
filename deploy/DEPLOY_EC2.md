# EC2 Deployment Guide (UFC Web Predictor)

## Recommended EC2 Specs

- **Recommended (production):** `m7i.2xlarge` (8 vCPU, 32 GiB RAM, x86_64)
- **Budget (low traffic):** `m7i.xlarge` (4 vCPU, 16 GiB RAM, x86_64)
- **Disk:** gp3 EBS, 80-120 GiB
- **OS:** Ubuntu 22.04 LTS (x86_64)

Why this sizing:
- The app fits and trains multiple `max_power` models at startup.
- Tree ensembles and in-memory dataset transforms benefit from higher RAM.
- x86_64 avoids Python wheel compatibility surprises (especially for Torch).

## Security Group

- Inbound: `22` (your IP only), `80` (0.0.0.0/0), `443` (0.0.0.0/0)
- Outbound: default allow

## 1) Provision Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx git
```

## 2) Place Project on Host

```bash
sudo mkdir -p /opt/ufc-elf
sudo chown -R ubuntu:ubuntu /opt/ufc-elf
cd /opt/ufc-elf
# copy or git clone your project contents here
```

## 3) Python Environment

```bash
cd /opt/ufc-elf
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 4) Systemd Service

```bash
sudo cp deploy/systemd/ufc-elf.service /etc/systemd/system/ufc-elf.service
sudo systemctl daemon-reload
sudo systemctl enable ufc-elf
sudo systemctl start ufc-elf
sudo systemctl status ufc-elf --no-pager
```

Note: `deploy/systemd/ufc-elf.service` assumes `User=ubuntu`. Change it if needed.

## 5) Nginx Reverse Proxy

```bash
sudo cp deploy/nginx/ufc-elf.conf /etc/nginx/sites-available/ufc-elf
sudo ln -sf /etc/nginx/sites-available/ufc-elf /etc/nginx/sites-enabled/ufc-elf
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 6) Verify

```bash
curl http://127.0.0.1:8000/healthz
curl http://<EC2_PUBLIC_IP>/healthz
```

## 7) Optional TLS (Recommended)

Use Certbot after DNS is pointing to the instance:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Model/Data Operations

- The app caches trained models in `data/model_cache/` and reuses them across restarts/deploys.
- Use the web UI:
  - `Update Data (Run Scraper)` to ingest new UFCStats data.
  - `Retrain Models` to rebuild model cache from updated CSV.
