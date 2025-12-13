# Visualization Module for Whisper Benchmark

## Generated Plots

1. **wer_comparison.png** - Word Error Rate across models (bar chart)
2. **cer_comparison.png** - Character Error Rate across models (bar chart)
3. **latency_comparison.png** - Latency metrics (mean, P95, P99)
4. **rtf_comparison.png** - Real-Time Factor with threshold line
5. **accuracy_vs_speed.png** - Scatter plot showing trade-offs
6. **metrics_heatmap.png** - All metrics in one heatmap
7. **wer_distributions.png** - WER distribution per model (histograms)
8. **summary_table.png** - Results summary as a table image

## Usage

### Automatic

Visualizations are generated automatically when running the benchmark:

```bash
cd .. && ./run_benchmark.sh
```

or 

```bash
python benchmark_whisper.py
```

### Manual

To regenerate plots from existing results:

```bash
cd /mloscratch/users/teissier/MediTalk/benchmark/whisper
python -m visualization.visualize
```
