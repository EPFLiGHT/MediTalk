#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Simple restart script for MediTalk services
# Usage: ./scripts/restart.sh [service_name] to restart a specific service
#        ./restart.sh to restart all services

"$SCRIPT_DIR/stop-local.sh" "$@"
"$SCRIPT_DIR/start-local.sh" "$@"
