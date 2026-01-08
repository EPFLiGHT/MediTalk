"""
Shared Data Sampler - USM dataset sampling for benchmarks

Creates stratified or random samples from United-Syn-Med dataset.
Generates unified samples for both ASR and TTS benchmarks to ensure consistency.

This module provides centralized data sampling logic to ensure both ASR and TTS
benchmarks use identical source data, guaranteeing fair comparisons.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import librosa

logger = logging.getLogger(__name__)


def create_benchmark_sample(
    metadata_path: str,
    sample_size: int,
    output_path: Optional[str] = None,
    strategy: str = "stratified",
    seed: int = 42,
    benchmark_type: str = "unified",
    audio_base_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create benchmark sample from USM dataset.
    
    Args:
        metadata_path: Path to USM metadata.csv
        sample_size: Number of samples to select (must be > 0)
        output_path: Optional path to save sample CSV
        strategy: 'random' or 'stratified' (stratified by text length)
        seed: Random seed for reproducibility
        benchmark_type: Output format - 'asr', 'tts', or 'unified' (default: unified)
        audio_base_path: Base path to resolve relative audio paths (defaults to metadata parent dir)
        
    Returns:
        DataFrame with sampled data formatted for the specified benchmark type
        
    Raises:
        FileNotFoundError: If metadata_path doesn't exist
        ValueError: If required columns are missing or sample_size is invalid
    """
    metadata_path = Path(metadata_path).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    if audio_base_path is None:
        audio_base_path = metadata_path
        while audio_base_path.name != 'MediTalk' and audio_base_path.parent != audio_base_path:
            audio_base_path = audio_base_path.parent
        if audio_base_path.name != 'MediTalk':
            raise ValueError(f"Could not find MediTalk root from {metadata_path}")
    else:
        audio_base_path = Path(audio_base_path).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    if sample_size <= 0:
        raise ValueError(f"sample_size must be > 0, got {sample_size}")
    
    if strategy not in ["random", "stratified"]:
        raise ValueError(f"strategy must be 'random' or 'stratified', got '{strategy}'")
    
    if benchmark_type not in ["asr", "tts", "unified"]:
        raise ValueError(f"benchmark_type must be 'asr', 'tts', or 'unified', got '{benchmark_type}'")
    
    logger.info(f"Loading metadata from {metadata_path}")
    logger.info(f"Audio base path: {audio_base_path}")
    
    # Load metadata (in chunks if large)
    try:
        df = pd.read_csv(metadata_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Metadata file is empty: {metadata_path}")
    except Exception as e:
        logger.warning(f"Standard loading failed, trying chunked loading: {e}")
        try:
            chunks = []
            for chunk in pd.read_csv(metadata_path, chunksize=50000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        except Exception as chunk_error:
            raise ValueError(f"Failed to load metadata file: {chunk_error}")
    
    logger.info(f"Loaded {len(df)} samples from metadata")
    
    # Validate required columns
    if df.empty:
        raise ValueError("Metadata DataFrame is empty after loading")
    
    if 'transcription' not in df.columns:
        raise ValueError(f"Column 'transcription' not found. Available columns: {list(df.columns)}")
    
    # Sample based on strategy
    if strategy == "stratified":
        sample_df = _stratified_sample(df, sample_size, seed)
    else:  # random
        sample_df = _random_sample(df, sample_size, seed)
    
    # Format output based on benchmark type
    sample_df = _format_output(sample_df, benchmark_type, audio_base_path)
    
    logger.info(f"Created {benchmark_type} sample with {len(sample_df)} items")
    
    if output_path:
        sample_df.to_csv(output_path, index=False)
        logger.info(f"Saved sample to {output_path}")
    
    return sample_df


def _format_output(df: pd.DataFrame, benchmark_type: str, audio_base_path: Path) -> pd.DataFrame:
    if benchmark_type == "asr":
        required_cols = ['audio_path', 'transcription']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for ASR format: {missing}")
        
        result = df[required_cols].copy()
        
        def get_duration(path):
            try:
                full_path = audio_base_path / path
                return librosa.get_duration(path=str(full_path))
            except Exception as e:
                raise ValueError(f"Could not get duration for audio file: {path}. Error: {e}")
        
        logger.info(f"Computing audio durations for {len(result)} samples...")
        result['audio_duration'] = result['audio_path'].apply(get_duration)
        
        logger.info(f"ASR format: {len(result)} samples with columns: {list(result.columns)}")
        return result
        
    elif benchmark_type == "tts":
        result = df.copy()
        if 'sample_id' not in result.columns:
            result.insert(0, 'sample_id', range(len(result)))
        result['text'] = result['transcription']
        result['text_length'] = result['text'].str.len()
        
        cols = ['sample_id', 'text', 'text_length']
        if 'audio_path' in result.columns:
            cols.append('audio_path')
            
            # Compute audio durations
            def get_duration(path):
                try:
                    full_path = audio_base_path / path
                    return librosa.get_duration(path=str(full_path))
                except Exception as e:
                    logger.warning(f"Could not get duration for {path}: {e}")
                    return None
            
            logger.info(f"Computing audio durations for {len(result)} samples...")
            result['audio_duration'] = result['audio_path'].apply(get_duration)
            cols.append('audio_duration')
            
        result = result[cols].copy()
        logger.info(f"TTS format: {len(result)} samples with columns: {cols}")
        
    else:  # unified
        result = df.copy()
        if 'sample_id' not in result.columns:
            result.insert(0, 'sample_id', range(len(result)))
        result['text'] = result['transcription']
        result['text_length'] = result['text'].str.len()
        logger.info(f"Unified format: {len(result)} samples with {len(result.columns)} columns")
    
    return result


def _random_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Simple random sampling."""
    if n >= len(df):
        logger.warning(f"Requested {n} samples but only {len(df)} available")
        return df.copy()
    
    return df.sample(n=n, random_state=seed)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified sampling by transcription length using quantile-based bins.
    Ensures representation across different text lengths with balanced bin sizes.
    
    Creates bins dynamically based on data distribution (quantiles),
    ensuring each bin has roughly equal representation in the dataset.
    
    Args:
        df: Input DataFrame with 'transcription' column
        n: Number of samples to select
        seed: Random seed for reproducibility
        
    Returns:
        Stratified sample DataFrame
    """
    df = df.copy()
    df['text_length'] = df['transcription'].str.len()
    
    # number of bins: (min 2, max 10, based on sample size)
    num_bins = min(10, max(2, n // 10))  # at least 10 samples per bin ideally
    
    # quantile-based bins for balanced distribution
    try:
        df['length_bin'] = pd.qcut(df['text_length'], q=num_bins, labels=False, duplicates='drop')
    except ValueError as e:
        raise ValueError(f"Error creating length bins: {e}")
    
    bin_counts = df.groupby('length_bin')['text_length'].agg(['count', 'min', 'max'])
    logger.info(f"Text length bins (quantile-based):")
    for idx, row in bin_counts.iterrows():
        logger.info(f"  Bin {idx}: {int(row['min'])}-{int(row['max'])} chars, {int(row['count'])} samples")
    
    unique_bins = sorted(df['length_bin'].dropna().unique())
    num_non_empty_bins = len(unique_bins)
    
    if num_non_empty_bins == 0:
        raise ValueError("No non-empty bins found for stratified sampling")
    
    # samples per bin with proper distribution
    samples_per_bin = n // num_non_empty_bins
    remainder = n % num_non_empty_bins  # extra samples to distribute
    
    sampled_dfs = []
    
    for idx, bin_label in enumerate(unique_bins):
        bin_df = df[df['length_bin'] == bin_label]
        # remainder distributed across first few bins
        extra = 1 if idx < remainder else 0
        sample_size = min(samples_per_bin + extra, len(bin_df))
        sampled = bin_df.sample(n=sample_size, random_state=seed + int(bin_label) * 100)
        sampled_dfs.append(sampled)
        logger.debug(f"Bin {bin_label}: sampled {sample_size}/{len(bin_df)} samples")
    
    if not sampled_dfs:
        raise ValueError("No samples could be drawn from any bin")
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    if len(result) != n:
        logger.warning(f"Stratified sampling produced {len(result)} samples instead of {n}")
        if len(result) < n:
            remaining = n - len(result)
            unused = df[~df.index.isin(result.index)]
            if len(unused) > 0:
                additional = unused.sample(n=min(remaining, len(unused)), random_state=seed + 1000)
                result = pd.concat([result, additional], ignore_index=True)
                logger.info(f"Added {len(additional)} random samples to reach target size")
        elif len(result) > n:
            result = result.iloc[:n]
    
    result = result.drop(columns=['text_length', 'length_bin'], errors='ignore')
    
    logger.info(f"Stratified sampling complete: {len(result)} samples across {num_non_empty_bins} bins")
    return result