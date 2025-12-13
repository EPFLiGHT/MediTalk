"""
Simple data sampling for benchmarking - supports random and stratified sampling.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_benchmark_sample(
    metadata_path: str, 
    sample_size: int, 
    output_path: str = None, 
    strategy: str = "random",
    seed: int = 42
):
    """
    Create a sample from USM dataset for benchmarking.
    
    Args:
        metadata_path: Path to USM metadata.csv
        sample_size: Number of samples to select
        output_path: Optional path to save the sample
        strategy: 'random' or 'stratified' (by transcription length)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled data
    """
    logger.info(f"Loading metadata from {metadata_path}")
    
    # Load metadata in chunks (file is large)
    chunks = []
    for chunk in pd.read_csv(metadata_path, chunksize=50000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    
    logger.info(f"Loaded {len(df)} samples, using {strategy} sampling")
    
    # Sample based on strategy
    if strategy == "stratified":
        sample_df = _stratified_sample(df, sample_size, seed)
    else:  # random
        sample_df = _random_sample(df, sample_size, seed)
    
    # Save if requested
    if output_path:
        sample_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(sample_df)} samples to {output_path}")
    
    return sample_df


def _random_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Simple random sampling."""
    if n >= len(df):
        logger.warning(f"Requested {n} samples but only {len(df)} available")
        return df.copy()
    
    return df.sample(n=n, random_state=seed)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified sampling by transcription length.
    Ensures representation across short, medium, and long transcriptions.
    """
    # Add length column
    df = df.copy()
    df['text_length'] = df['transcription'].str.len()
    
    # Create bins: 0-50, 50-100, 100-200, 200-500, 500+
    bins = [0, 50, 100, 200, 500, float('inf')]
    df['length_bin'] = pd.cut(df['text_length'], bins=bins, labels=False, include_lowest=True)
    
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
    
    # Clean up temporary columns
    result = result.drop(columns=['text_length', 'length_bin'], errors='ignore')
    
    logger.info(f"Stratified sampling: selected {len(result)} samples across {len(bins)} length bins")
    return result
