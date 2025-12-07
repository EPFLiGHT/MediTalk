#!/bin/bash
# GPU Monitoring Script for MediTalk
# Shows real-time GPU memory usage with service mapping

watch -n 1 '
echo "┌────────────────── GPU Memory Usage ───────────────────────┐"
printf "│ %-4s │ %-12s │ %-8s │ %-6s │ %-6s │ %-6s │\n" "GPU" "Memory Used" "Total" "Free" "Util%" "Temp°C"
echo "├──────┼──────────────┼──────────┼────────┼────────┼────────┤"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx used total free util temp; do
  printf "│ %-4s │ %8s MiB │ %8s │ %6s │ %5s%% │ %5s°C│\n" "$idx" "$used" "$total" "$free" "$util" "$temp"
done
echo "└──────┴──────────────┴──────────┴────────┴────────┴────────┘"
echo ""
echo "Last updated: $(date +"%H:%M:%S")"
'
