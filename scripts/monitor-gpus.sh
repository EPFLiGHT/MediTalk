#!/bin/bash
# GPU Monitoring Script for MediTalk
# Shows real-time GPU memory usage with service mapping

watch -n 1 '
echo "┌─────────────────────────────────────────────────────────────────────────────────────┐"
echo "│                          MediTalk GPU Monitoring                                    │"
echo "├─────────────────────────────────────────────────────────────────────────────────────┤"
echo "│ GPU Allocation:                                                                     │"
echo "│   • GPU 0-1-2: General services (Orpheus, Meditron, Bark, CSM, Whisper)              │"
echo "│   • GPU 3:   (Reserved/Available for future services)                              │"
echo "└─────────────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─── GPU Memory Usage ────────────────────────────────────────────────────────────────┐"
printf "│ %-4s │ %-12s │ %-10s │ %-10s │ %-6s │ %-6s │\n" "GPU" "Memory Used" "Total" "Free" "Util%" "Temp°C"
echo "├──────┼──────────────┼────────────┼────────────┼────────┼────────┤"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx used total free util temp; do
  printf "│ %-4s │ %8s MiB │ %8s │ %8s │ %5s%% │ %5s°C│\n" "$idx" "$used" "$total" "$free" "$util" "$temp"
done
echo "└──────┴──────────────┴────────────┴────────────┴────────┴────────┘"
echo ""
echo "┌─── Active Processes by GPU ─────────────────────────────────────────────────────────┐"
printf "│ %-8s │ %-12s │ %-50s │\n" "PID" "Memory (MiB)" "GPU Bus ID"
echo "├──────────┼──────────────┼────────────────────────────────────────────────────────┤"
nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader,nounits | while IFS=, read -r pid mem bus; do
  # Map bus ID to GPU index
  case "$bus" in
    *03:00.0*) gpu="GPU 0" ;;
    *23:00.0*) gpu="GPU 1" ;;
    *43:00.0*) gpu="GPU 2" ;;
    *63:00.0*) gpu="GPU 3" ;;
    *83:00.0*) gpu="GPU 4" ;;
    *A3:00.0*|*a3:00.0*) gpu="GPU 5" ;;
    *) gpu="$bus" ;;
  esac
  printf "│ %-8s │ %10s   │ %-50s │\n" "$pid" "$mem" "$gpu"
done
echo "└──────────┴──────────────┴────────────────────────────────────────────────────────┘"
echo ""
echo "Last updated: $(date +"%H:%M:%S")"
'
