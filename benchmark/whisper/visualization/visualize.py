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
    """Plot WER comparison with 95% confidence intervals."""
    models = list(results.keys())
    wer_means = [results[m]['wer']['mean'] for m in models]
    wer_ci_lower = [results[m]['wer']['ci_lower'] for m in models]
    wer_ci_upper = [results[m]['wer']['ci_upper'] for m in models]
    wer_errors = [
        [wer_means[i] - wer_ci_lower[i] for i in range(len(models))],
        [wer_ci_upper[i] - wer_means[i] for i in range(len(models))]
    ]
    
    fig, ax = plt.subplots()
    colors = ['#10b981' if wer < 0.15 else '#f59e0b' if wer < 0.30 else '#ef4444' 
              for wer in wer_means]
    
    bars = ax.bar(models, wer_means, color=colors, alpha=0.8, edgecolor='black',
                  yerr=wer_errors, capsize=5, error_kw={'elinewidth': 2, 'alpha': 0.7})
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Word Error Rate (WER)')
    ax.set_title('Word Error Rate with 95% CI - Lower is Better')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cer_comparison(results, output_dir):
    """Plot CER comparison with 95% confidence intervals."""
    models = list(results.keys())
    cer_means = [results[m]['cer']['mean'] for m in models]
    cer_ci_lower = [results[m]['cer']['ci_lower'] for m in models]
    cer_ci_upper = [results[m]['cer']['ci_upper'] for m in models]
    cer_errors = [
        [cer_means[i] - cer_ci_lower[i] for i in range(len(models))],
        [cer_ci_upper[i] - cer_means[i] for i in range(len(models))]
    ]
    
    fig, ax = plt.subplots()
    colors = ['#10b981' if cer < 0.08 else '#f59e0b' if cer < 0.15 else '#ef4444' 
              for cer in cer_means]
    
    bars = ax.bar(models, cer_means, color=colors, alpha=0.8, edgecolor='black',
                  yerr=cer_errors, capsize=5, error_kw={'elinewidth': 2, 'alpha': 0.7})
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Character Error Rate (CER)')
    ax.set_title('Character Error Rate with 95% CI - Lower is Better')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_comparison(results, output_dir):
    """Plot latency with confidence intervals and percentiles."""
    models = list(results.keys())
    lat_means = [results[m]['latency']['mean'] for m in models]
    lat_ci_lower = [results[m]['latency']['ci_lower'] for m in models]
    lat_ci_upper = [results[m]['latency']['ci_upper'] for m in models]
    lat_errors = [
        [lat_means[i] - lat_ci_lower[i] for i in range(len(models))],
        [lat_ci_upper[i] - lat_means[i] for i in range(len(models))]
    ]
    
    fig, ax = plt.subplots()
    bars = ax.bar(models, lat_means, color='#667eea', alpha=0.8, edgecolor='black',
                  yerr=lat_errors, capsize=5, error_kw={'elinewidth': 2, 'alpha': 0.7},
                  label='Mean with 95% CI')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency with 95% Confidence Interval - Lower is Better')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rtf_comparison(results, output_dir):
    """Plot Real-Time Factor with confidence intervals."""
    models = list(results.keys())
    rtf_means = [results[m]['rtf']['mean'] for m in models]
    rtf_ci_lower = [results[m]['rtf']['ci_lower'] for m in models]
    rtf_ci_upper = [results[m]['rtf']['ci_upper'] for m in models]
    rtf_errors = [
        [rtf_means[i] - rtf_ci_lower[i] for i in range(len(models))],
        [rtf_ci_upper[i] - rtf_means[i] for i in range(len(models))]
    ]
    
    fig, ax = plt.subplots()
    colors = ['#10b981' if rtf < 1.0 else '#ef4444' for rtf in rtf_means]
    bars = ax.bar(models, rtf_means, color=colors, alpha=0.8, edgecolor='black',
                  yerr=rtf_errors, capsize=5, error_kw={'elinewidth': 2, 'alpha': 0.7})
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time threshold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Real-Time Factor (RTF)')
    ax.set_title('RTF with 95% CI - Lower is Better\n(< 1.0 = faster than real-time)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
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
    
    for idx, model in enumerate(models[:4]):
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
            model.upper(),
            f"{results[model]['wer']['mean']:.4f}",
            f"{results[model]['cer']['mean']:.4f}",
            f"{results[model]['latency']['mean']:.2f}",
            f"{results[model]['rtf']['mean']:.3f}",
            str(results[model]['sample_count'])
        ]
        table_data.append(row)
    
    fig, ax = plt.subplots(figsize=(12, len(models) * 0.8 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'WER', 'CER', 'Latency\n(mean, s)', 'RTF', 'Samples'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.18, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        bg_color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(6):
            table[(i, j)].set_facecolor(bg_color)
    
    best_wer_idx = min(range(len(models)), key=lambda i: results[models[i]]['wer']['mean']) + 1
    table[(best_wer_idx, 1)].set_facecolor('#c6f6d5')
    table[(best_wer_idx, 1)].set_text_props(weight='bold')
    
    best_cer_idx = min(range(len(models)), key=lambda i: results[models[i]]['cer']['mean']) + 1
    table[(best_cer_idx, 2)].set_facecolor('#c6f6d5')
    table[(best_cer_idx, 2)].set_text_props(weight='bold')
    
    plt.title('Whisper Benchmark Results Summary\n(Green cells = best performance)', 
             fontsize=14, fontweight='bold', pad=10)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_percentiles(results, output_dir):
    """Plot latency percentiles (mean, p95, p99) to show tail latencies."""
    models = list(results.keys())
    lat_means = [results[m]['latency']['mean'] for m in models]
    lat_p95 = [results[m]['latency']['p95'] for m in models]
    lat_p99 = [results[m]['latency']['p99'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, lat_means, width, label='Mean', color='#667eea', alpha=0.8)
    ax.bar(x, lat_p95, width, label='P95', color='#f59e0b', alpha=0.8)
    ax.bar(x + width, lat_p99, width, label='P99', color='#ef4444', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency Percentiles - Tail Latency Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rtf_percentiles(results, output_dir):
    """Plot RTF percentiles (mean, p95, p99)."""
    models = list(results.keys())
    rtf_means = [results[m]['rtf']['mean'] for m in models]
    rtf_p95 = [results[m]['rtf']['p95'] for m in models]
    rtf_p99 = [results[m]['rtf']['p99'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, rtf_means, width, label='Mean', color='#10b981', alpha=0.8)
    ax.bar(x, rtf_p95, width, label='P95', color='#f59e0b', alpha=0.8)
    ax.bar(x + width, rtf_p99, width, label='P99', color='#ef4444', alpha=0.8)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Real-time threshold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Real-Time Factor (RTF)')
    ax.set_title('RTF Percentiles - Performance Consistency Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtf_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_uncertainty_comparison(results, output_dir):
    """Plot confidence interval ranges for all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results.keys())
    metrics = [
        ('wer', 'WER', axes[0, 0], 'Error Rate'),
        ('cer', 'CER', axes[0, 1], 'Error Rate'),
        ('latency', 'Latency', axes[1, 0], 'Time (s)'),
        ('rtf', 'RTF', axes[1, 1], 'Time Factor')
    ]
    
    for metric, label, ax, ylabel in metrics:
        means = [results[m][metric]['mean'] for m in models]
        ci_lower = [results[m][metric]['ci_lower'] for m in models]
        ci_upper = [results[m][metric]['ci_upper'] for m in models]
        
        sort_idx = np.argsort(means)
        sorted_models = [models[i] for i in sort_idx]
        sorted_means = [means[i] for i in sort_idx]
        sorted_lower = [ci_lower[i] for i in sort_idx]
        sorted_upper = [ci_upper[i] for i in sort_idx]
        
        y_pos = np.arange(len(sorted_models))
        
        for i, (model, mean, lower, upper) in enumerate(zip(sorted_models, sorted_means, sorted_lower, sorted_upper)):
            ax.plot([lower, upper], [i, i], 'o-', linewidth=2.5, markersize=6, alpha=0.7)
            ax.plot(mean, i, 'D', markersize=8, color='black', zorder=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_models)
        ax.set_xlabel(ylabel)
        ax.set_title(f'{label} with 95% Confidence Intervals')
        ax.grid(axis='x', alpha=0.3)
        
        ax.plot([], [], 'D', markersize=8, color='black', label='Mean')
        ax.plot([], [], 'o-', linewidth=2.5, markersize=6, alpha=0.7, label='95% CI')
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_ranges(results, output_dir):
    """Plot min-mean-median-max ranges for key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results.keys())
    metrics = [
        ('wer', 'WER', axes[0, 0], 'Error Rate'),
        ('cer', 'CER', axes[0, 1], 'Error Rate'),
        ('latency', 'Latency', axes[1, 0], 'Time (s)'),
        ('rtf', 'RTF', axes[1, 1], 'Time Factor')
    ]
    
    for metric, label, ax, ylabel in metrics:
        means = [results[m][metric]['mean'] for m in models]
        mins = [results[m][metric]['min'] for m in models]
        maxs = [results[m][metric]['max'] for m in models]
        medians = [results[m][metric]['median'] for m in models]
        
        x_pos = np.arange(len(models))
        
        for i, (model, mean, median, min_val, max_val) in enumerate(zip(models, means, medians, mins, maxs)):
            ax.plot([i, i], [min_val, max_val], 'o-', linewidth=2, markersize=5, alpha=0.5, color='gray')
            ax.plot(i, mean, 's', markersize=10, color='blue', zorder=5, label='Mean' if i == 0 else '')
            ax.plot(i, median, '^', markersize=10, color='orange', zorder=5, label='Median' if i == 0 else '')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{label} Performance Range (Min-Mean-Median-Max)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_ranges.png', dpi=300, bbox_inches='tight')
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
    
    print("Loading benchmark results...")
    results = load_results(results_dir)
    print(f"   Found {len(results)} models\n")
    
    plots = [
        ("WER Comparison", plot_wer_comparison),
        ("CER Comparison", plot_cer_comparison),
        ("Latency Comparison", plot_latency_comparison),
        ("Latency Percentiles", plot_latency_percentiles),
        ("RTF Comparison", plot_rtf_comparison),
        ("RTF Percentiles", plot_rtf_percentiles),
        ("Accuracy vs Speed", plot_accuracy_vs_speed),
        ("Metrics Heatmap", plot_metrics_heatmap),
        ("Uncertainty Comparison", plot_uncertainty_comparison),
        ("Performance Ranges", plot_performance_ranges),
        ("Summary Table", plot_summary_table),
    ]
    
    for name, plot_func in plots:
        try:
            print(f"Generating {name}...")
            if plot_func == plot_wer_distribution:
                plot_func(results, results_dir, output_dir)
            else:
                plot_func(results, output_dir)
            print(f"   Saved to plots/")
        except Exception as e:
            print(f"   Failed: {e}")
    
    try:
        print(f"Generating WER Distributions...")
        plot_wer_distribution(results, results_dir, output_dir)
        print(f"   Saved to plots/")
    except Exception as e:
        print(f"   Skipped (detailed results not available): {e}")
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "../results"
    
    generate_all_plots(results_dir)
