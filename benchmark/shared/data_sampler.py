"""
Shared Data Sampler - USM dataset sampling for benchmarks

Creates stratified or random samples from United-Syn-Med dataset.

Used by:
- benchmark/whisper/ (ASR evaluation with audio + transcription)
- benchmark/tts/ (TTS evaluation with transcription only)
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_benchmark_sample(
    metadata_path: str,
    sample_size: int,
    output_path: str = None,
    strategy: str = "stratified",
    seed: int = 42,
    text_column: str = "transcription"
) -> pd.DataFrame:
    """
    Create benchmark sample from USM dataset.
    
    Args:
        metadata_path: Path to USM metadata.csv
        sample_size: Number of samples to select
        output_path: Optional path to save sample
        strategy: 'random' or 'stratified'
        seed: Random seed for reproducibility
        text_column: Name of text column (default: 'transcription')
        
    Returns:
        DataFrame with sampled data
    """
    logger.info(f"Loading metadata from {metadata_path}")
    
    # Load metadata (in chunks if large)
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        # Try chunked loading for very large files
        logger.info(f"Large file, loading in chunks...")
        chunks = []
        for chunk in pd.read_csv(metadata_path, chunksize=50000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    
    logger.info(f"Loaded {len(df)} samples")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
    
    # Sample based on strategy
    if strategy == "stratified":
        sample_df = _stratified_sample(df, sample_size, seed, text_column)
    else:  # random
        sample_df = _random_sample(df, sample_size, seed)
    
    logger.info(f"Created sample with {len(sample_df)} items")
    
    if output_path:
        sample_df.to_csv(output_path, index=False)
        logger.info(f"Saved sample to {output_path}")
    
    return sample_df


def _random_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Simple random sampling."""
    if n >= len(df):
        logger.warning(f"Requested {n} samples but only {len(df)} available")
        return df.copy()
    
    return df.sample(n=n, random_state=seed)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int, text_column: str) -> pd.DataFrame:
    """
    Stratified sampling by text length.
    Bins: very-short (0-50), short (50-100), medium (100-200), long (200-500), very-long (500+)
    """
    df = df.copy()
    df['text_length'] = df[text_column].str.len()
    
    # Create length bins
    bins = [0, 50, 100, 200, 500, float('inf')]
    df['length_bin'] = pd.cut(
        df['text_length'],
        bins=bins,
        labels=['very-short', 'short', 'medium', 'long', 'very-long'],
        include_lowest=True
    )
    
    # Sample equally from each bin
    samples_per_bin = n // len(bins)
    sampled_dfs = []
    
    for bin_idx in df['length_bin'].unique():
        if pd.isna(bin_idx):
            continue
        bin_df = df[df['length_bin'] == bin_idx]
        sample_size = min(samples_per_bin, len(bin_df))
        sampled = bin_df.sample(n=sample_size, random_state=seed + int(bin_idx))
        sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we need more samples, add random ones
    if len(result) < n:
        remaining = n - len(result)
        unused = df[~df.index.isin(result.index)]
        if len(unused) > 0:
            additional = unused.sample(n=min(remaining, len(unused)), random_state=seed + 1000)
            result = pd.concat([result, additional], ignore_index=True)
    
    result = result.drop(columns=['text_length', 'length_bin'], errors='ignore')
    
    logger.info(f"Stratified sampling: selected {len(result)} samples across {len(bins)} length bins")
    return result
