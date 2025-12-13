# Whisper ASR Benchmark

Benchmark for evaluating Whisper ASR models on medical speech (USM dataset).

## Usage

```bash
./run_benchmark.sh
```

The script will:
1. Install dependencies
2. Prompt you to configure Whisper with each model size
3. Run benchmarks for all models
4. Generate results and visualizations

## Results

Results saved to `results/`:
- `model_comparison.json` - Summary of all models
- `{model}_detailed_results.csv` - Per-sample results for each model
- `{model}_summary.json` - Metrics for each model
- `plots/` - 8 visualization charts (cf. visualization/README.md)

## Configuration

Edit `config.yaml`:

```yaml
models:
  sizes: [tiny, base, small, medium, large]  # Models to test

data:
  sample_size: 1000  # Number of samples (max 790'865)
  sampling_strategy: "stratified"  # or "random"
```

## Workflow

For each model in your config:
1. Script pauses and asks you to prepare Whisper service
2. You restart Whisper service with the right model size
3. Press Enter in the benchmark script
4. Benchmark runs for that model
5. Repeat for next model

## Requirements


- Get the USM dataset (cf. `../data/fetching/README.md`
- Process the USM dataset (cf. `../data/processing/README.md`))

## Metrics

- **WER** (Word Error Rate): Lower is better (< 0.10 excellent)
- **CER** (Character Error Rate): Lower is better
- **Latency**: Processing time per file
- **RTF** (Real-Time Factor): < 1.0 = faster than real-time
