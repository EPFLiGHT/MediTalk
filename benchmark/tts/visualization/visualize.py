"""
Visualization script for TTS benchmark results.
Generates comprehensive plots for evaluating TTS model performance.

Metrics visualized:
- WER (Word Error Rate) - Lower is better
- CER (Character Error Rate) - Lower is better  
- RTF (Real-Time Factor) - Lower is better, <1.0 means faster than real-time
- MOS (Mean Opinion Score) - Higher is better (1-5 scale), measures perceived quality
- Generation Time - Time to synthesize audio
- Audio Duration - Length of generated audio
- Success Rate - Percentage of successful syntheses
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def load_results(results_dir):
    """Load benchmark results from model_comparison.json."""
    results_dir = Path(results_dir)
    comparison_file = results_dir / "model_comparison.json"
    
    if not comparison_file.exists():
        raise FileNotFoundError(f"No results found at {comparison_file}")
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)
    
    # Filter out error entries (entries without a 'model' field)
    valid_data = [entry for entry in data if 'model' in entry]
    
    # Convert list to DataFrame for easier manipulation
    df = pd.DataFrame(valid_data)
    return df


def plot_wer_cer_comparison(df, output_dir):
    """Plot WER and CER comparison side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = df['model'].values
    wer_means = df['wer_mean'].values
    cer_means = df['cer_mean'].values
    
    # WER plot - color by performance zones
    colors_wer = ['#10b981' if wer < 0.15 else '#f59e0b' if wer < 0.30 else '#ef4444' 
                  for wer in wer_means]
    bars1 = ax1.bar(models, wer_means, color=colors_wer, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Word Error Rate (WER)')
    ax1.set_title('Word Error Rate - Lower is Better')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # WER legend
    wer_legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#10b981', ec='black', alpha=0.8, label='Excellent (< 0.15)'),
        Rectangle((0, 0), 1, 1, fc='#f59e0b', ec='black', alpha=0.8, label='Acceptable (0.15-0.30)'),
        Rectangle((0, 0), 1, 1, fc='#ef4444', ec='black', alpha=0.8, label='Poor (≥ 0.30)'),
    ]
    ax1.legend(handles=wer_legend_elements, loc='upper right', fontsize=8, framealpha=0.95)
    
    # CER plot - color by performance zones
    colors_cer = ['#10b981' if cer < 0.08 else '#f59e0b' if cer < 0.15 else '#ef4444' 
                  for cer in cer_means]
    bars2 = ax2.bar(models, cer_means, color=colors_cer, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Character Error Rate (CER)')
    ax2.set_title('Character Error Rate - Lower is Better')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # CER legend
    cer_legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#10b981', ec='black', alpha=0.8, label='Excellent (< 0.08)'),
        Rectangle((0, 0), 1, 1, fc='#f59e0b', ec='black', alpha=0.8, label='Acceptable (0.08-0.15)'),
        Rectangle((0, 0), 1, 1, fc='#ef4444', ec='black', alpha=0.8, label='Poor (≥ 0.15)'),
    ]
    ax2.legend(handles=cer_legend_elements, loc='upper right', fontsize=8, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wer_cer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mos_comparison(df, output_dir):
    """Plot MOS (Mean Opinion Score) comparison if available."""
    # Check if MOS data exists
    if 'mos_mean' not in df.columns or df['mos_mean'].isna().all():
        print("   Skipping - No MOS data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].values
    mos_means = df['mos_mean'].values
    mos_stds = df.get('mos_std', pd.Series([0]*len(models))).values
    
    # Color based on MOS score (green for high, red for low)
    colors = sns.color_palette("RdYlGn", len(models))
    # Sort by MOS to apply colors correctly
    sorted_indices = np.argsort(mos_means)
    bar_colors = [colors[i] for i in sorted_indices]
    
    bars = ax.bar(models, mos_means, yerr=mos_stds, color=bar_colors, 
                  alpha=0.8, edgecolor='black', capsize=5)
    
    ax.set_ylabel('Predicted Mean Opinion Score (MOS)')
    ax.set_title('Predicted Mean Opinion Score - Higher is Better (1-5 scale) \n (Predicted using NISQA-TTS)')
    ax.set_ylim(1, 5)  # MOS scale is 1-5
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add horizontal reference lines
    ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.3, label='Acceptable (3.0)')
    ax.axhline(y=4.0, color='green', linestyle='--', alpha=0.3, label='Good (4.0)')
    ax.legend(loc='upper left')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mos_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rtf_comparison(df, output_dir):
    """Plot Real-Time Factor with real-time threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].values
    rtf_means = df['rtf_mean'].values
    rtf_medians = df['rtf_median'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    # Color bars based on whether they're faster than real-time
    colors_mean = ['#10b981' if rtf < 1.0 else '#f59e0b' if rtf < 2.0 else '#ef4444' 
                   for rtf in rtf_means]
    colors_median = ['#059669' if rtf < 1.0 else '#d97706' if rtf < 2.0 else '#dc2626' 
                     for rtf in rtf_medians]
    
    # Mean: solid bars, Median: hatched bars for visual distinction
    bars1 = ax.bar(x - width/2, rtf_means, width, label='Mean', 
                   color=colors_mean, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, rtf_medians, width, label='Median', 
                   color=colors_median, alpha=0.8, edgecolor='black', hatch='//')
    
    # Add real-time threshold line
    threshold_line = ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Real-Time Factor (RTF)')
    ax.set_title('Real-Time Factor Comparison - Lower is Better\n(RTF < 1.0 = faster than real-time)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    legend_elements = [
        threshold_line,
        Rectangle((0, 0), 1, 1, fc='#10b981', ec='black', alpha=0.8, label='Fast (RTF < 1.0)'),
        Rectangle((0, 0), 1, 1, fc='#f59e0b', ec='black', alpha=0.8, label='Acceptable (1.0 ≤ RTF < 2.0)'),
        Rectangle((0, 0), 1, 1, fc='#ef4444', ec='black', alpha=0.8, label='Slow (RTF ≥ 2.0)'),
        Rectangle((0, 0), 1, 1, fc='gray', ec='black', alpha=0.8, label='Mean'),
        Rectangle((0, 0), 1, 1, fc='gray', ec='black', alpha=0.8, hatch='//', label='Median'),
    ]
    
    ax.legend(handles=legend_elements, 
             labels=['Real-time threshold (RTF=1.0)', 
                    'Fast (RTF < 1.0)', 
                    'Acceptable (1.0 ≤ RTF < 2.0)', 
                    'Slow (RTF ≥ 2.0)',
                    'Mean',
                    'Median'],
             loc='upper left', fontsize=9, framealpha=0.95)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rtf_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_quality_vs_speed(df, output_dir):
    """Scatter plot: TTS Quality (WER) vs Speed (RTF)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    models = df['model'].values
    wer_means = df['wer_mean'].values
    rtf_means = df['rtf_mean'].values
    
    # Size points by CER (smaller CER = larger point)
    sizes = 1000 / (df['cer_mean'].values + 0.01)  # Add small constant to avoid division by zero

    # add legend for sizes 
    ax.scatter([], [], s=300, c='gray', alpha=0.6, label=f'Point Size ~ 1/CER \n (Larger point = Lower CER)')
    ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Point Size Legend')
    
    scatter = ax.scatter(rtf_means, wer_means, s=sizes, alpha=0.6, 
                        c=range(len(models)), cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for i, model in enumerate(models):
        x_offset = len(model) * -3
        ax.annotate(model, (rtf_means[i], wer_means[i]), 
                   xytext=(x_offset, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0))
    
    # Add reference lines
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Real-time (RTF=1.0)')
    
    ax.set_xlabel('Real-Time Factor (RTF)')
    ax.set_ylabel('Word Error Rate (WER)')
    ax.set_title('TTS Quality vs Speed Trade-off\n(Lower-left corner is ideal: low WER, low RTF)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Invert axes to make "better" be top-right
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_vs_speed.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_generation_metrics(df, output_dir):
    """Plot generation time and audio duration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = df['model'].values
    gen_times = df['avg_generation_time'].values
    audio_durations = df['avg_audio_duration'].values
    
    # Generation time plot
    colors = sns.color_palette("YlOrRd", len(models))
    bars1 = ax1.bar(models, gen_times, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Average Generation Time (seconds)')
    ax1.set_title('Average Time to Synthesize Audio')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)
    
    # Audio duration plot
    colors2 = sns.color_palette("Greens", len(models))
    bars2 = ax2.bar(models, audio_durations, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Audio Duration (seconds)')
    ax2.set_title('Average Generated Audio Length')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_radar(df, output_dir):
    """Radar/spider chart comparing all normalized metrics across models."""
    from math import pi
    
    models = df['model'].values
    
    # Select metrics to compare (normalize to 0-1, invert where lower is better)
    metrics = ['WER', 'CER', 'RTF', 'Gen Time']
    
    # Normalize metrics (invert so higher is better for visualization)
    wer_norm = 1 - (df['wer_mean'] / df['wer_mean'].max())
    cer_norm = 1 - (df['cer_mean'] / df['cer_mean'].max())
    rtf_norm = 1 - (df['rtf_mean'] / df['rtf_mean'].max())
    gen_norm = 1 - (df['avg_generation_time'] / df['avg_generation_time'].max())
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Plot each model
    colors = sns.color_palette("husl", len(models))
    
    for idx, model in enumerate(models):
        values = [wer_norm.iloc[idx], cer_norm.iloc[idx], 
                 rtf_norm.iloc[idx], gen_norm.iloc[idx]]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(), 
               color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=11)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=9)
    ax.grid(True)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Normalized Performance Metrics\n(Higher = Better)', 
             size=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_radar.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_heatmap(df, output_dir):
    """Heatmap of all metrics across models."""
    models = df['model'].values
    
    metrics_data = {
        'WER': df['wer_mean'].values,
        'CER': df['cer_mean'].values,
        'RTF': df['rtf_mean'].values,
        'Gen Time (s)': df['avg_generation_time'].values,
        'Audio Dur (s)': df['avg_audio_duration'].values,
    }
    
    metrics_df = pd.DataFrame(metrics_data, index=models)
    
    # Normalize for heatmap (0-1 scale)
    df_norm = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(df_norm, annot=metrics_df.values, fmt='.3f', 
                cmap='RdYlGn_r', cbar_kws={'label': 'Normalized Score (0=best, 1=worst)'},
                linewidths=1, linecolor='white', ax=ax,
                vmin=0, vmax=1)
    
    ax.set_title('Metrics Heatmap - All Models\n(Cells show actual values, colors show normalized scores)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax.set_ylabel('Models', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_variability(df, output_dir):
    """Plot WER with error bars showing standard deviation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].values
    wer_means = df['wer_mean'].values
    wer_stds = df['wer_std'].values
    cer_means = df['cer_mean'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    # WER with error bars
    bars1 = ax.bar(x - width/2, wer_means, width, yerr=wer_stds, 
                   label='WER (with std dev)', color='#667eea', 
                   alpha=0.8, edgecolor='black', capsize=5)
    
    # CER without error bars for comparison
    bars2 = ax.bar(x + width/2, cer_means, width,
                   label='CER', color='#f093fb',
                   alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate Variability\n(WER shown with standard deviation error bars)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_variability.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_table(df, output_dir):
    """Create a comprehensive summary table as an image."""
    models = df['model'].values
    has_mos = 'mos_mean' in df.columns and not df['mos_mean'].isna().all()
    
    table_data = []
    for idx, row in df.iterrows():
        table_row = [
            row['model'].upper(),
            f"{row['wer_mean']:.4f}",
            f"{row['cer_mean']:.4f}",
            f"{row['rtf_mean']:.3f}",
        ]
        if has_mos:
            table_row.append(f"{row.get('mos_mean', 0):.2f}")
        table_row.extend([
            f"{row['avg_generation_time']:.1f}s",
            f"{row['success_rate']*100:.0f}%",
            str(row['total_samples'])
        ])
        table_data.append(table_row)
    
    fig, ax = plt.subplots(figsize=(14 if has_mos else 12, len(models) * 0.8 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ['Model', 'WER', 'CER', 'RTF']
    if has_mos:
        col_labels.append('MOS')
    col_labels.extend(['Mean\nGen Time', 'Success', 'Samples'])
    
    col_widths = [0.15, 0.10, 0.10, 0.10]
    if has_mos:
        col_widths.append(0.10)
    col_widths.extend([0.12, 0.10, 0.10])
    
    table = ax.table(cellText=table_data,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    num_cols = len(col_labels)
    for i in range(num_cols):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        # Alternate row colors
        bg_color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(num_cols):
            table[(i, j)].set_facecolor(bg_color)
    
    # Highlight best/worst performers

    # Best WER (column 1)
    best_wer_idx = df['wer_mean'].idxmin() + 1
    table[(best_wer_idx, 1)].set_facecolor('#c6f6d5')
    table[(best_wer_idx, 1)].set_text_props(weight='bold')
    # Worst WER (column 1)
    worst_wer_idx = df['wer_mean'].idxmax() + 1
    table[(worst_wer_idx, 1)].set_facecolor('#fed7dd')
    table[(worst_wer_idx, 1)].set_text_props(weight='bold')
    
    # Best CER (column 2)
    best_cer_idx = df['cer_mean'].idxmin() + 1
    table[(best_cer_idx, 2)].set_facecolor('#c6f6d5')
    table[(best_cer_idx, 2)].set_text_props(weight='bold')
    # Worst CER (column 2)
    worst_cer_idx = df['cer_mean'].idxmax() + 1
    table[(worst_cer_idx, 2)].set_facecolor('#fed7dd')
    table[(worst_cer_idx, 2)].set_text_props(weight='bold')
    
    # Best RTF (column 3)
    best_rtf_idx = df['rtf_mean'].idxmin() + 1
    table[(best_rtf_idx, 3)].set_facecolor('#c6f6d5')
    table[(best_rtf_idx, 3)].set_text_props(weight='bold')
    # Worst RTF (column 3)
    worst_rtf_idx = df['rtf_mean'].idxmax() + 1
    table[(worst_rtf_idx, 3)].set_facecolor('#fed7dd')
    table[(worst_rtf_idx, 3)].set_text_props(weight='bold')
    
    # Best MOS (column 4 if present)
    if has_mos:
        best_mos_idx = df['mos_mean'].idxmax() + 1
        table[(best_mos_idx, 4)].set_facecolor('#c6f6d5')
        table[(best_mos_idx, 4)].set_text_props(weight='bold')
    # Worst MOS (column 4 if present)
    if has_mos:
        worst_mos_idx = df['mos_mean'].idxmin() + 1
        table[(worst_mos_idx, 4)].set_facecolor('#fed7dd')
        table[(worst_mos_idx, 4)].set_text_props(weight='bold')
    
    plt.title('TTS Benchmark Results Summary\n(Green cells = best performance, Red cells = worst performance)', 
             fontsize=14, fontweight='bold', pad=5)
    
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_rankings(df, output_dir):
    """Create a ranking visualization showing how models rank across different metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df['model'].values
    metrics = ['WER', 'CER', 'RTF', 'Gen Time']
    
    # Calculate rankings (1 = best, n = worst)
    rankings = {
        'WER': df['wer_mean'].rank().values,
        'CER': df['cer_mean'].rank().values,
        'RTF': df['rtf_mean'].rank().values,
        'Gen Time': df['avg_generation_time'].rank().values,
    }
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    colors = sns.color_palette("Set2", len(models))
    
    for idx, model in enumerate(models):
        ranks = [rankings[metric][idx] for metric in metrics]
        offset = (idx - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, ranks, width, label=model.upper(), 
                     color=colors[idx], alpha=0.8, edgecolor='black')
        
        # Add rank labels
        for bar, rank in zip(bars, ranks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(rank)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Rank (1 = Best)')
    ax.set_title('Model Rankings Across Metrics\n(Lower rank = Better performance)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(1, len(models) + 1))
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.invert_yaxis()  # Invert so rank 1 is at top
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_plots(results_dir, output_dir=None):
    """Generate all visualization plots for TTS benchmark."""
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_dir / 'plots'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("  TTS Benchmark Visualization")
    print(f"{'='*70}\n")
    
    # Load results
    print("Loading benchmark results...")
    try:
        df = load_results(results_dir)
        print(f"   Found {len(df)} models: {', '.join(df['model'].values)}\n")
    except FileNotFoundError as e:
        print(f"   ✗ Error: {e}")
        return None
    
    # Generate each plot
    plots = [
        ("WER & CER Comparison", plot_wer_cer_comparison),
        ("MOS Comparison", plot_mos_comparison),
        ("RTF Comparison", plot_rtf_comparison),
        ("Quality vs Speed Trade-off", plot_quality_vs_speed),
        ("Generation Metrics", plot_generation_metrics),
        ("Metrics Radar Chart", plot_metrics_radar),
        ("Metrics Heatmap", plot_metrics_heatmap),
        ("Error Variability", plot_error_variability),
        ("Model Rankings", plot_model_rankings),
        ("Summary Table", plot_summary_table),
    ]
    
    successful = 0
    failed = 0
    
    for name, plot_func in plots:
        try:
            print(f"Generating {name}...")
            plot_func(df, output_dir)
            print(f"   ✓ Saved")
            successful += 1
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"✓ Visualization complete")
    print(f"   {successful} plots generated successfully")
    if failed > 0:
        print(f"   {failed} plots failed")
    print(f"   Location: {output_dir}")
    print(f"{'='*70}\n")
    
    return output_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "../results"
    
    generate_all_plots(results_dir)
