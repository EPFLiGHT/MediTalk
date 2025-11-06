# MediTalk Scripts

This folder contains all the operational scripts for MediTalk.

## Usage

```bash
# From project root:
cd /mloscratch/users/teissier/MediTalk

# Start services
./scripts/start-local.sh

# Check health
./scripts/health-check.sh

# Monitor GPUs
./scripts/monitor-gpus.sh

# Stop services
./scripts/stop-local.sh

# Restart all or specific service
./scripts/restart.sh
./scripts/restart.sh modelOrpheus
```

## Scripts

### Service Management

- **start-local.sh** - Start all MediTalk services
- **stop-local.sh** - Stop all running MediTalk services
- **restart.sh** - Restart all or specific services

### Setup & Monitoring

- **setup-local.sh** - Initial setup (install dependencies, create .env)
- **health-check.sh** - Check the status of all services (ports, processes, logs)
- **monitor-gpus.sh** - Real-time GPU memory monitoring with service mapping