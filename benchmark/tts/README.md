# TTS Benchmark

Automated benchmarking suite for Text-to-Speech models on medical speech (USM dataset).

## Usage

```bash
./run_benchmark.sh
```

By default, benchmarks all TTS models in parallel (faster, requires all services running). Set `parallel_models: false` in config for sequential mode with prompts. Results are saved to `results/` with automated visualizations.

## Configuration

Edit `config.yaml` to customize benchmark parameters:

```yaml
dataset:
  metadata_path: "../../data/processed/USM/metadata.csv"
  sample_size: 1000
  sampling_strategy: "stratified"  # or "random"

tts_models:
  - name: "orpheus"
    url: "http://localhost:5005"
  - name: "bark"
    url: "http://localhost:5008"
  - name: "csm"
    url: "http://localhost:5010"
  - name: "qwen3omni"
    url: "http://localhost:5014"

whisper_service:
  url: "http://localhost:5007"

nisqa_service:
  enabled: true  # Set to false to skip MOS evaluation
  url: "http://localhost:8006"
```

## Metrics

The benchmark evaluates TTS models using the following metrics:

- **WER/CER**: Word and character error rates via ASR round-trip (Whisper transcription)
- **RTF**: Real-time factor (generation speed, lower is better)
- **MOS**: Mean opinion score for quality assessment (NISQA-TTS, 1-5 scale, higher is better)
- **Latency**: Processing time per sample
- **Confidence Intervals**: 95% CI, min/max ranges for all metrics

## Output

Results are saved to timestamped directories in `results/`:

```
results/
├── benchmark_sample.csv           # Dataset sample used
├── model_comparison.json          # Aggregated metrics with CI
├── YYYYMMDD_HHMMSS_modelname/     # Per-model results
│   ├── detailed_results.csv
│   ├── summary.json
│   └── audio_samples/
└── plots/                         # Visualization charts
    ├── wer_cer_comparison.png
    ├── rtf_comparison.png
    ├── mos_comparison.png
    ├── uncertainty_comparison.png
    ├── performance_ranges.png
    └── summary_table.png
```

## Requirements

- Get the USM dataset (cf. `../data/fetching/README.md`)
- Process the USM dataset (cf. `../data/processing/README.md`)

- Running TTS model services
- Running Whisper ASR service (http://localhost:5007)
- Running NISQA service (http://localhost:8006)