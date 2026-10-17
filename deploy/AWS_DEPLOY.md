# Deploying Audible Insights on AWS (EC2 + Docker)

The app runs as one Docker container on a small EC2 instance. A start-up script (`deploy/ec2_user_data.sh`) does
all the server work: you click through the EC2 launch screen once, wait about 8 minutes, and the app is live at
`http://<instance-public-ip>`.

```
Browser ──HTTP :80──▶ EC2 t3.micro (Amazon Linux 2023)
                        └─ Docker container "audible-insights"
                             └─ Streamlit :8501  (code + trained recommender baked into the image)
```

**Cost.** On a new AWS account's Free Plan, a t3.micro is paid for out of the sign-up credits, so you are not
charged. It runs at roughly $8–11 a month including its public IP. **Stop or terminate the instance when you no
longer need it** (step 6).

---

## 0. Before you start
1. The repository must be **public on GitHub**, because the instance clones it. If your repo is not
   `aryaman2213/audible-insights`, edit `REPO_URL` at the top of `deploy/ec2_user_data.sh` first.
2. An AWS account. Sign in to the console and pick the region **Asia Pacific (Mumbai) ap-south-1** (top-right).

## 1. Launch the instance
EC2 → **Launch instance**, then fill in:

| Setting | Value |
|---|---|
| Name | `audible-insights` |
| AMI | **Amazon Linux 2023** (the default, "Free tier eligible") |
| Instance type | **t3.micro** (or t2.micro: whichever is marked free-tier eligible) |
| Key pair | *Proceed without a key pair* is fine (you can still connect from the browser with EC2 Instance Connect), or create one if you prefer SSH |
| Network settings → Firewall | Create security group, tick **Allow HTTP traffic from the internet**. Leave SSH allowed only from *My IP*. |
| Storage | 10 GiB gp3 |
| Advanced details → **User data** | paste the whole of `deploy/ec2_user_data.sh` |

Click **Launch instance**.

## 2. Wait for it to build (about 8 minutes)
The script installs Docker, clones the repo, builds the image and starts the container. Open the instance and
copy its **Public IPv4 address**, then visit `http://<that-address>` (plain **http**, not https).

If the page doesn't load after 10 minutes: Instance → **Connect** → *EC2 Instance Connect* → Connect, then run
```bash
sudo tail -n 40 /var/log/cloud-init-output.log
sudo docker ps
```
The container should show as `Up … (healthy)`.

## 3. (Optional) a fixed address
The public IP changes if you stop and start the instance. For a permanent address: EC2 → **Elastic IPs** →
Allocate → Associate it with the instance. An Elastic IP is free while attached to a *running* instance, and
billed when it isn't.

## 4. Deploying a new version
Push to GitHub, connect to the instance (Instance Connect), then run:
```bash
sudo bash /opt/audible-insights/deploy/update.sh
```

## 5. Accessibility and scalability (for the evaluation)
- **Accessible:** public HTTP on port 80. The container restarts on its own after crashes or reboots
  (`--restart unless-stopped`), and Docker health-checks Streamlit's `/_stcore/health` endpoint.
- **Stateless:** the trained recommender is baked into the image and nothing is written at runtime. That means the
  app scales horizontally without code changes:
  - **More capacity on one machine:** change the instance type (Actions → Instance settings → Change instance
    type).
  - **Many machines:** create an AMI from this instance, put it in an **Auto Scaling group** behind an
    **Application Load Balancer** (target port 80, health check path `/_stcore/health`, enable sticky sessions,
    because Streamlit keeps a websocket per user).
  - **Managed alternative:** **Elastic Beanstalk** → Create application → Platform *Docker* → upload a zip of the
    repo. Beanstalk reads the `Dockerfile`, maps port 80 to the container's 8501, and handles the load balancer and
    auto scaling for you.
- **Optional S3 storage:** the brief mentions keeping processed data in S3. It is small (2 MB) and built into
  the image, so S3 isn't needed. To use it, upload `data/processed/` and `models/` to a bucket and download them
  at container start with `aws s3 cp`.

## 6. Stop paying
- **Pause:** Instance state → **Stop**. Compute charges stop; the 10 GiB disk costs a few cents a month.
- **Remove completely:** Instance state → **Terminate**, and release any Elastic IP.
- **Set a safety net:** Billing → Budgets → create a *zero-spend* or $5 budget with email alerts.
