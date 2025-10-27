cd /mloscratch/users/teissier

/mloscratch/users/teissier

kubernetes > rcp-caas-prod > Workloads > Pods > right click on meditron-basic-0-0 > Attach to VSCode

# Web UI logs
tail -f logs/webui.log

# Meditron AI logs
tail -f logs/modelMeditron.log

# Orpheus TTS logs
tail -f logs/modelOrpheus.log

# Whisper ASR logs
tail -f logs/modelWhisper.log

# Monitor Memory Usage
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# try streamlit
