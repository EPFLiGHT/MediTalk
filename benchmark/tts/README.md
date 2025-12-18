# TTS Benchmark

Automated benchmarking suite for Text-to-Speech models on medical conversational speech.

## Usage

```bash
./run_benchmark.sh
```

The benchmark will prompt you to start each TTS service sequentially. Results are saved to `results/` with automated visualizations.

## Configuration

Edit `config.yaml` to customize benchmark parameters:

```yaml
dataset:
  metadata_path: "../../data/processed/dailytalk/metadata.csv"
  sample_size: 100
  sampling_strategy: "random"

tts_models:
  - name: "orpheus"
    url: "http://localhost:5005"
  - name: "bark"
    url: "http://localhost:5008"

nisqa_service:
  enabled: true
  url: "http://localhost:8006"
```

## Metrics

The benchmark evaluates TTS models using the following metrics:

- **WER/CER**: Word and character error rates via ASR round-trip (Whisper transcription)
- **RTF**: Real-time factor (generation speed, lower is better)
- **MOS**: Mean opinion score for quality assessment (NISQA-TTS, 1-5 scale)

## Output

Results are saved to timestamped directories in `results/`:

```
results/
├── benchmark_sample.csv           # Dataset sample used
├── model_comparison.json          # Aggregated metrics
├── YYYYMMDD_HHMMSS_modelname/     # Per-model results
│   ├── detailed_results.csv
│   ├── summary.json
│   └── audio_samples/
└── plots/                         # Visualization charts
    ├── wer_cer_comparison.png
    ├── rtf_comparison.png
    ├── mos_comparison.png
    └── summary_table.png
```

## Requirements

- Running TTS model services
- Whisper ASR service (http://localhost:8000)
- NISQA service (http://localhost:8006)