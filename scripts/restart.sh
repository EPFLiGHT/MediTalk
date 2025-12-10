#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_NAME="$1"

declare -A SERVICES=(
    ["modelOrpheus"]="Orpheus TTS"
    ["modelBark"]="Bark TTS"
    ["modelCSM"]="CSM TTS"
    ["modelWhisper"]="Whisper ASR"
    ["modelMultiMeditron"]="MultiMeditron AI"
    ["modelQwen3Omni"]="Qwen3-Omni AI"
    ["webui"]="FastAPI WebUI"
    ["webui-streamlit"]="Streamlit WebUI"
    ["controller"]="Controller Service"
)

usage() {
    echo "Usage: $0 [service_name]"
    echo ""
    echo "Restart MediTalk services (stop + start)"
    echo ""
    echo "Arguments:"
    echo "  service_name    Optional. Name of specific service to restart."
    echo "                  If omitted, all services will be restarted."
    echo ""
    echo "Available services:"
    for service in "${!SERVICES[@]}"; do
        echo "  - $service (${SERVICES[$service]})"
    done | sort
    echo ""
    echo "Examples:"
    echo "  $0                      # Restart all services"
    echo "  $0 modelOrpheus         # Restart only Orpheus service"
    echo "  $0 webui-streamlit      # Restart only Streamlit UI"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    exit 0
}

# Show help if requested
if [ "$SERVICE_NAME" == "-h" ] || [ "$SERVICE_NAME" == "--help" ]; then
    usage
fi

# Validate service name if provided
if [ ! -z "$SERVICE_NAME" ] && [ -z "${SERVICES[$SERVICE_NAME]}" ]; then
    echo "ERROR: Unknown service '$SERVICE_NAME'"
    echo ""
    echo "Available services:"
    for service in "${!SERVICES[@]}"; do
        echo "  - $service (${SERVICES[$service]})"
    done | sort
    echo ""
    echo "Use --help for more information"
    exit 1
fi

# Display what we're doing
if [ -z "$SERVICE_NAME" ]; then
    echo "Restarting all MediTalk services..."
else
    echo "Restarting ${SERVICES[$SERVICE_NAME]} ($SERVICE_NAME)..."
fi
echo ""

# Stop then start the service(s)
"$SCRIPT_DIR/stop-local.sh" "$@"
sleep 2
"$SCRIPT_DIR/start-local.sh" "$@"
