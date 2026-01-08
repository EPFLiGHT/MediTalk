#!/usr/bin/env python3
"""Compute statistics for ASR and TTS benchmark samples."""

import pandas as pd
import numpy as np
import re
from pathlib import Path


def compute_statistics(df, text_column):
    """Compute text statistics from DataFrame."""
    texts = df[text_column].fillna('').astype(str)
    
    # Word and character counts
    word_counts = texts.str.split().str.len()
    char_counts = texts.str.len()
    
    # Unique vocabulary
    all_words = ' '.join(texts.str.lower()).split()
    unique_vocab = len(set(re.findall(r'\b[a-z0-9]+\b', ' '.join(all_words))))
    
    # Audio duration (if available in CSV)
    audio_durations = df['audio_duration'] if 'audio_duration' in df.columns else None
    
    return word_counts, char_counts, unique_vocab, audio_durations


def format_stats(data, decimals=2):
    """Format statistics for LaTeX."""
    mean_sd = f"{np.mean(data):.{decimals}f} $\\pm$ {np.std(data, ddof=1):.{decimals}f}"
    median_range = f"{np.median(data):.{decimals}f} ({np.min(data):.{decimals}f}--{np.max(data):.{decimals}f})"
    return mean_sd, median_range


def print_latex_table(name, n_samples, word_counts, char_counts, unique_vocab, audio_durations=None):
    """Print LaTeX table."""
    print(f"\n{'='*70}")
    print(f"{name} ({n_samples:,} samples)")
    print("="*70)
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Dataset characteristics for the {n_samples:,} evaluated samples}}")
    print("\\label{tab:dataset_stats}")
    print("\\setlength{\\tabcolsep}{8pt}")
    print("\\renewcommand{\\arraystretch}{1.25}")
    print("\\begin{tabular}{l c c}")
    print("\\toprule")
    print("\\textbf{Characteristic} & \\textbf{Mean $\\pm$ SD} & \\textbf{Median (Range)} \\\\")
    print("\\midrule")
    
    if audio_durations is not None and len(audio_durations) > 0:
        mean_sd, median_range = format_stats(audio_durations, 2)
        print(f"Audio duration (seconds) & {mean_sd} & {median_range} \\\\")
    else:
        print(f"Audio duration (seconds) & N/A & N/A \\\\")
    
    mean_sd, median_range = format_stats(word_counts, 1)
    print(f"Transcription word count & {mean_sd} & {median_range} \\\\")
    
    mean_sd, median_range = format_stats(char_counts, 1)
    print(f"Transcription character count & {mean_sd} & {median_range} \\\\")
    
    # Format unique vocab with proper comma separator
    vocab_formatted = f"{unique_vocab:,}".replace(",", "{,}")
    print(f"Unique vocabulary terms & \\multicolumn{{2}}{{c}}{{{vocab_formatted}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Main function to compute and display statistics."""
    benchmark_dir = Path(__file__).parent
    
    whisper_csv = benchmark_dir / "whisper" / "results" / "benchmark_sample.csv"
    tts_csv = benchmark_dir / "tts" / "results" / "benchmark_sample.csv"
    
    # ASR benchmark
    if whisper_csv.exists():
        print("\nProcessing ASR (Whisper) benchmark...")
        df = pd.read_csv(whisper_csv)
        word_counts, char_counts, unique_vocab, audio_durations = compute_statistics(df, 'transcription')
        print_latex_table("ASR (Whisper)", len(df), word_counts, char_counts, unique_vocab, audio_durations)
    else:
        print(f"Not found: {whisper_csv}")
    
    # TTS benchmark
    if tts_csv.exists():
        print("\nProcessing TTS benchmark...")
        df = pd.read_csv(tts_csv)
        word_counts, char_counts, unique_vocab, audio_durations = compute_statistics(df, 'text')
        print_latex_table("TTS", len(df), word_counts, char_counts, unique_vocab, audio_durations)
    else:
        print(f"Not found: {tts_csv}")


if __name__ == "__main__":
    main()
