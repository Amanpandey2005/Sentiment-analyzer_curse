# Production Deployment Guide

## 1. Provision AWS Infrastructure

Create or select an EC2 key pair, then run Terraform:

```bash
cd terraform
terraform init
terraform apply \
  -var="allowed_ssh_cidr=YOUR_IP/32" \
  -var="ec2_key_name=YOUR_KEY_PAIR"
```

Terraform creates:

- ECR repository for Docker images
- EC2 instance with an IAM instance profile
- S3 bucket for future model artifacts
- CloudWatch log group
- Security group for SSH, HTTP, HTTPS, Grafana, and Prometheus

## 2. Prepare EC2

SSH into the instance and clone the repository:

```bash
sudo mkdir -p /opt/ml-sentiment-platform
sudo chown -R ubuntu:ubuntu /opt/ml-sentiment-platform
git clone https://github.com/YOUR_ORG/YOUR_REPO.git /opt/ml-sentiment-platform
cd /opt/ml-sentiment-platform
```

Create `/opt/ml-sentiment-platform/.env` for production values:

```bash
AWS_ACCOUNT_ID=123456789012
AWS_REGION=us-east-1
ECR_REPOSITORY=ml-sentiment-platform
DATABASE_URL=postgresql+psycopg://user:password@host:5432/sentiment
REDIS_URL=redis://host:6379/0
JWT_SECRET_KEY=replace-with-a-long-random-secret
GRAFANA_ADMIN_PASSWORD=replace-me
```

For a compact portfolio deployment you can run Postgres and Redis on the same EC2 host by extending `docker-compose.prod.yml`, but managed RDS and ElastiCache are preferred for production.

## 3. Configure GitHub Secrets

Add these repository secrets:

- `AWS_ACCOUNT_ID`
- `AWS_REGION`
- `AWS_GITHUB_ACTIONS_ROLE_ARN`
- `ECR_REPOSITORY`
- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`

The GitHub Actions role needs ECR push permissions. EC2 needs permissions to pull ECR images and write CloudWatch logs.

## 4. Enable HTTPS

Place certificates on EC2:

```bash
mkdir -p /opt/ml-sentiment-platform/nginx/certs
cp fullchain.pem /opt/ml-sentiment-platform/nginx/certs/fullchain.pem
cp privkey.pem /opt/ml-sentiment-platform/nginx/certs/privkey.pem
cp nginx/https.conf.example nginx/https.conf
```

You can generate certificates with Certbot or use ACM with an AWS load balancer in front of EC2.

## 5. Deploy

Push to `main`:

```bash
git push origin main
```

GitHub Actions will run tests, lint, build, push to ECR, SSH to EC2, pull the latest image, and restart containers.

## 6. Observability

- API: `http://EC2_PUBLIC_IP/api/health`
- Dashboard: `http://EC2_PUBLIC_IP`
- Prometheus: `http://EC2_PUBLIC_IP:9090`
- Grafana: `http://EC2_PUBLIC_IP:3000`
- CloudWatch logs: `/${project_name}/api`
