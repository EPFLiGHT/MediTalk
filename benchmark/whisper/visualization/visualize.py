"""
Simple visualization script for Whisper benchmark results.
Generates all plots automatically and saves to results/plots/
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_results(results_dir):
    """Load all benchmark results."""
    results_dir = Path(results_dir)
    comparison_file = results_dir / "model_comparison.json"
    
    if not comparison_file.exists():
        raise FileNotFoundError(f"No results found at {comparison_file}")
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)
    
    return data['comparison']


def plot_wer_comparison(results, output_dir):
    """Plot WER comparison across models."""
    models = list(results.keys())
    wer_means = [results[m]['wer']['mean'] for m in models]
    wer_medians = [results[m]['wer']['median'] for m in models]
    
    fig, ax = plt.subplots()
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, wer_means, width, label='Mean', color='#667eea')
    ax.bar(x + width/2, wer_medians, width, label='Median', color='#764ba2')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Word Error Rate (WER)')
    ax.set_title('Word Error Rate Comparison - Lower is Better')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cer_comparison(results, output_dir):
    """Plot CER comparison across models."""
    models = list(results.keys())
    cer_means = [results[m]['cer']['mean'] for m in models]
    
    fig, ax = plt.subplots()
    colors = sns.color_palette("viridis", len(models))
    ax.bar(models, cer_means, color=colors)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Character Error Rate (CER)')
    ax.set_title('Character Error Rate Comparison - Lower is Better')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_comparison(results, output_dir):
    """Plot latency metrics across models."""
    models = list(results.keys())
    lat_means = [results[m]['latency']['mean'] for m in models]
    lat_p95 = [results[m]['latency']['p95'] for m in models]
    lat_p99 = [results[m]['latency']['p99'] for m in models]
    
    fig, ax = plt.subplots()
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, lat_means, width, label='Mean', color='#10b981')
    ax.bar(x, lat_p95, width, label='P95', color='#f59e0b')
    ax.bar(x + width, lat_p99, width, label='P99', color='#ef4444')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency Comparison - Lower is Better')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rtf_comparison(results, output_dir):
    """Plot Real-Time Factor comparison."""
    models = list(results.keys())
    rtf_means = [results[m]['rtf']['mean'] for m in models]
    
    fig, ax = plt.subplots()
    colors = ['#10b981' if rtf < 1.0 else '#ef4444' for rtf in rtf_means]
    ax.bar(models, rtf_means, color=colors)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time threshold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Real-Time Factor (RTF)')
    ax.set_title('Real-Time Factor - Lower is Better (< 1.0 = faster than real-time)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtf_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_speed(results, output_dir):
    """Scatter plot: WER vs Latency."""
    models = list(results.keys())
    wer_means = [results[m]['wer']['mean'] for m in models]
    lat_p95 = [results[m]['latency']['p95'] for m in models]
    
    fig, ax = plt.subplots()
    ax.scatter(lat_p95, wer_means, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        ax.annotate(model, (lat_p95[i], wer_means[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Latency P95 (seconds)')
    ax.set_ylabel('Word Error Rate (WER)')
    ax.set_title('Accuracy vs Speed Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_speed.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_heatmap(results, output_dir):
    """Heatmap of all metrics across models."""
    models = list(results.keys())
    
    metrics_data = {
        'WER': [results[m]['wer']['mean'] for m in models],
        'CER': [results[m]['cer']['mean'] for m in models],
        'Latency (s)': [results[m]['latency']['p95'] for m in models],
        'RTF': [results[m]['rtf']['mean'] for m in models],
    }
    
    df = pd.DataFrame(metrics_data, index=models)
    
    # Normalize for heatmap
    df_norm = (df - df.min()) / (df.max() - df.min())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_norm, annot=df.values, fmt='.3f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Normalized Score'}, ax=ax)
    
    ax.set_title('Metrics Heatmap (annotated with actual values)')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_wer_distribution(results, results_dir, output_dir):
    """Plot WER distribution for each model."""
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, model in enumerate(models[:4]):  # Max 4 models
        detailed_file = results_dir / f"{model}_detailed_results.csv"
        if not detailed_file.exists():
            continue
        
        df = pd.read_csv(detailed_file)
        
        axes[idx].hist(df['wer'], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        axes[idx].axvline(df['wer'].mean(), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {df["wer"].mean():.3f}')
        axes[idx].set_xlabel('Word Error Rate')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{model.upper()} - WER Distribution')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(models), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wer_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_table(results, output_dir):
    """Create a summary table as an image."""
    models = list(results.keys())
    
    table_data = []
    for model in models:
        row = [
            model,
            f"{results[model]['wer']['mean']:.4f}",
            f"{results[model]['cer']['mean']:.4f}",
            f"{results[model]['latency']['p95']:.2f}",
            f"{results[model]['rtf']['mean']:.3f}",
            str(results[model]['sample_count'])
        ]
        table_data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'WER', 'CER', 'Latency P95', 'RTF', 'Samples'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Benchmark Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_plots(results_dir, output_dir=None):
    """Generate all visualization plots."""
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_dir / 'plots'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("  Generating Visualizations")
    print(f"{'='*60}\n")
    
    # Load results
    print("📊 Loading benchmark results...")
    results = load_results(results_dir)
    print(f"   Found {len(results)} models\n")
    
    # Generate each plot
    plots = [
        ("WER Comparison", plot_wer_comparison),
        ("CER Comparison", plot_cer_comparison),
        ("Latency Comparison", plot_latency_comparison),
        ("RTF Comparison", plot_rtf_comparison),
        ("Accuracy vs Speed", plot_accuracy_vs_speed),
        ("Metrics Heatmap", plot_metrics_heatmap),
        ("Summary Table", plot_summary_table),
    ]
    
    for name, plot_func in plots:
        try:
            print(f"📈 Generating {name}...")
            if plot_func == plot_wer_distribution:
                plot_func(results, results_dir, output_dir)
            else:
                plot_func(results, output_dir)
            print(f"   ✅ Saved to plots/")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # WER distribution (needs detailed results)
    try:
        print(f"📈 Generating WER Distributions...")
        plot_wer_distribution(results, results_dir, output_dir)
        print(f"   ✅ Saved to plots/")
    except Exception as e:
        print(f"   ⚠️  Skipped (detailed results not available): {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "../results"
    
    generate_all_plots(results_dir)
