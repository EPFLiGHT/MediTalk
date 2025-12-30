"""
TTS Benchmark

Runs the complete TTS benchmark pipeline.

Usage:
    python benchmark_tts.py --config config.yaml
"""

import argparse
import logging
import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


from data_sampler import create_benchmark_sample
from tts_client import TTSClient
from asr_client import ASRClient
from nisqa_client import NISQAClient
from visualization.visualize import generate_all_plots

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
import metrics as shared_metrics

calculate_wer = shared_metrics.calculate_wer
calculate_cer = shared_metrics.calculate_cer
calculate_rtf = shared_metrics.calculate_rtf


def setup_model_logger(model_name: str, results_dir: Path):
    """
    Setup a logger for a specific model (used in worker processes).
    
    Args:
        model_name: Name of the model
        results_dir: Directory to save log files
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(f"{model_name}_worker")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        f'%(asctime)s - [{model_name}] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    log_file = results_dir / f"{model_name}_benchmark.log"
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def _benchmark_model_worker(model_config: dict, data_sample_dict: dict, config: dict) -> dict:
    """
    Worker function to benchmark a single TTS model.
    
    Args:
        model_config: {'name': str, 'url': str}
        data_sample_dict: DataFrame converted to dict (for pickling)
        config: Full configuration dictionary from config.yaml
        
    Returns:
        Dictionary with aggregated results
    """

    # Setup:
    data_sample = pd.DataFrame(data_sample_dict)
    
    model_name = model_config['name']
    model_url = model_config['url']
    results_dir = Path(config['results_dir']).resolve()
    whisper_url = config['whisper_service']['url']
    nisqa_config = config.get('nisqa_service', None)
    save_audio = config.get('save_audio', True)
    
    logger = setup_model_logger(model_name, results_dir)
    
    logger.info(f"{'='*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*60}")
    
    tts_client = TTSClient(model_name, model_url)
    asr_client = ASRClient(whisper_url)
    
    nisqa_client = None
    if nisqa_config and nisqa_config.get('enabled', False):
        nisqa_client = NISQAClient(nisqa_config['url'])
    
    logger.info("Running health checks...")
    if not tts_client.health_check():
        logger.error(f"✗ {model_name} service is not responding!")
        return {
            'model': model_name,
            'error': f'{model_name} service not available',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    if not asr_client.health_check():
        logger.error("✗ Whisper ASR service is not responding!")
        return {
            'model': model_name,
            'error': 'Whisper service not available',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    if nisqa_client:
        if not nisqa_client.health_check():
            logger.warning("/!\ NISQA service not available - MOS scores will be skipped")
            nisqa_client = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_results_dir = results_dir / f"{timestamp}_{model_name}"
    audio_dir = model_results_dir / "audio_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {model_results_dir}")
    
    results = []
    
    # Process each sample:
    logger.info(f"Processing {len(data_sample)} samples...")
    for idx, row in tqdm(data_sample.iterrows(), total=len(data_sample), desc=f"{model_name}"):
        sample_id = row['sample_id']
        text = row['text']
        
        # Generate audio and evaluate
        audio_path = audio_dir / f"sample_{sample_id:04d}.wav"
        
        # Generate audio
        tts_result = tts_client.synthesize(text, str(audio_path))
        
        if not tts_result['success']:
            result_entry = {
                'sample_id': sample_id,
                'text': text,
                'error': tts_result['error'],
                'success': False
            }
            results.append(result_entry)
            continue
        
        # Transcribe audio (ASR round-trip)
        asr_result = asr_client.transcribe_file(str(audio_path))
        
        if not asr_result['success']:
            result_entry = {
                'sample_id': sample_id,
                'text': text,
                'audio_path': str(audio_path),
                'generation_time': tts_result['generation_time'],
                'audio_duration': tts_result['duration'],
                'error': asr_result['error'],
                'success': False
            }
            results.append(result_entry)
            continue
        
        # Calculate basic metrics
        transcription = asr_result['text']
        wer_score = calculate_wer(text, transcription)
        cer_score = calculate_cer(text, transcription)
        rtf = calculate_rtf(tts_result['generation_time'], tts_result['duration'])
        
        # Predict MOS quality score using NISQA
        mos_scores = {}
        if nisqa_client:
            nisqa_result = nisqa_client.predict_quality(str(audio_path))
            if nisqa_result['success']:
                mos_scores = {
                    'mos': nisqa_result['mos'],
                    'nisqa_latency': nisqa_result['latency']
                }
        
        result_entry = {
            'sample_id': sample_id,
            'text': text,
            'transcription': transcription,
            'audio_path': str(audio_path),
            'generation_time': tts_result['generation_time'],
            'audio_duration': tts_result['duration'],
            'rtf': rtf,
            'wer': wer_score,
            'cer': cer_score,
            'asr_latency': asr_result['latency'],
            'success': True
        }
        result_entry.update(mos_scores)
        results.append(result_entry)
    
    # Detailed results:
    results_df = pd.DataFrame(results)
    results_csv = model_results_dir / "detailed_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"✓ Saved detailed results to {results_csv}")
    
    # Summary statistics:
    successful = results_df[results_df['success'] == True]
    summary = {
        'model': model_name,
        'timestamp': timestamp,
        'total_samples': len(results_df),
        'successful_samples': len(successful),
        'failed_samples': len(results_df) - len(successful),
        'success_rate': len(successful) / len(results_df) if len(results_df) > 0 else 0,
    }
    
    if len(successful) > 0:
        summary.update({
            'wer_mean': float(successful['wer'].mean()),
            'wer_median': float(successful['wer'].median()),
            'wer_std': float(successful['wer'].std()),
            'cer_mean': float(successful['cer'].mean()),
            'cer_median': float(successful['cer'].median()),
            'rtf_mean': float(successful['rtf'].mean()),
            'rtf_median': float(successful['rtf'].median()),
            'avg_generation_time': float(successful['generation_time'].mean()),
            'avg_audio_duration': float(successful['audio_duration'].mean()),
            'total_audio_minutes': float(successful['audio_duration'].sum() / 60),
        })
        
        if 'mos' in successful.columns:
            summary.update({
                'mos_mean': float(successful['mos'].mean()),
                'mos_median': float(successful['mos'].median()),
                'mos_std': float(successful['mos'].std()),
            })
    
    summary_json = model_results_dir / "summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to {summary_json}\n")
    
    logger.info(f"{'='*60}")
    logger.info(f"RESULTS SUMMARY - {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Success Rate: {summary['success_rate']:.1%} ({summary['successful_samples']}/{summary['total_samples']})")
    if len(successful) > 0:
        logger.info(f"WER: {summary['wer_mean']:.3f} ± {summary['wer_std']:.3f} (median: {summary['wer_median']:.3f})")
        logger.info(f"CER: {summary['cer_mean']:.3f} (median: {summary['cer_median']:.3f})")
        logger.info(f"RTF: {summary['rtf_mean']:.3f} (median: {summary['rtf_median']:.3f})")
        if 'mos_mean' in summary:
            logger.info(f"MOS: {summary['mos_mean']:.2f} ± {summary['mos_std']:.2f} (median: {summary['mos_median']:.2f})")
        logger.info(f"Avg Generation Time: {summary['avg_generation_time']:.2f}s")
        logger.info(f"Total Audio: {summary['total_audio_minutes']:.1f} minutes")
    
    # Delete audio if not saving
    if not save_audio:
        import shutil
        shutil.rmtree(audio_dir)
        logger.info(f"✓ Cleaned up audio files (save_audio=False)")
    
    return summary


class TTSBenchmark:
    """TTS Benchmark runner with sequential and parallel execution support."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['results_dir']).resolve()
        self.results_dir.mkdir(exist_ok=True)
        
        self.whisper_url = self.config['whisper_service']['url']
        self.nisqa_config = self.config.get('nisqa_service', None)
        self.tts_models = self.config['tts_models']
        self.save_audio = self.config.get('save_audio', True)
        self.interactive = self.config.get('interactive_mode', True)
        self.parallel_models = self.config.get('parallel_models', False)
        self.max_parallel_models = self.config.get('max_parallel_models', None)
        
        # Cannot be interactive in parallel mode
        if self.parallel_models and self.interactive:
            logging.warning("Cannot use interactive_mode with parallel_models. Setting interactive_mode=False")
            self.interactive = False
    
    def run(self, data_sample: pd.DataFrame):
        """
        Run benchmark for all configured TTS models.
        Supports both sequential and parallel execution.
        
        Args:
            data_sample: DataFrame with text samples to benchmark
        """
        logger = logging.getLogger(__name__)
        
        if self.parallel_models:
            self._run_parallel(data_sample, logger)
        else:
            self._run_sequential(data_sample, logger)
    
    def _run_parallel(self, data_sample: pd.DataFrame, logger):
        """Run benchmarks in parallel."""
        logger.info("="*60)
        logger.info(f"PARALLEL MODE: Benchmarking {len(self.tts_models)} models concurrently")
        logger.info("Ensure all required services are running!")
        logger.info("="*60)
        
        data_sample_dict = data_sample.to_dict('list')
        
        all_summaries = []
        
        with ProcessPoolExecutor(max_workers=self.max_parallel_models) as executor:
            # Submit all jobs
            future_to_model = {
                executor.submit(
                    _benchmark_model_worker,
                    model_config,
                    data_sample_dict,
                    self.config
                ): model_config
                for model_config in self.tts_models
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_config = future_to_model[future]
                model_name = model_config['name']
                
                try:
                    summary = future.result()
                    all_summaries.append(summary)
                    
                    if 'error' in summary:
                        logger.error(f"✗ {model_name} failed: {summary['error']}")
                    else:
                        logger.info(f"✓ {model_name} completed successfully")
                        
                except Exception as e:
                    logger.error(f"✗ {model_name} raised exception: {e}")
                    all_summaries.append({
                        'model': model_name,
                        'error': str(e),
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    })
        
        # Save aggregated results
        self._save_comparison(all_summaries, logger)
        
        # Visualizations:
        self._generate_visualizations(logger)
        
        logger.info("="*60)
        logger.info("PARALLEL BENCHMARK COMPLETE!")
        logger.info("="*60)
    
    def _run_sequential(self, data_sample: pd.DataFrame, logger):
        """Run benchmarks sequentially (to use if not enough resources for parallel execution)."""
        logger.info("="*60)
        logger.info(f"SEQUENTIAL MODE: Benchmarking {len(self.tts_models)} models one by one")
        logger.info("="*60)
        
        all_summaries = []
        
        data_sample_dict = data_sample.to_dict('list')
        
        # Sequentially benchmark each model:
        for model_config in self.tts_models:
            model_name = model_config['name']
            
            logger.info(f"{'='*60}")
            logger.info(f"Ready to benchmark: {model_name}")
            logger.info(f"Service URL: {model_config['url']}")
            logger.info(f"{'='*60}")
            if self.interactive:
                input(f"\n-->  Please ensure {model_name} service is running, then press ENTER to continue...\n")
            
            # Run benchmark for this model
            summary = _benchmark_model_worker(model_config, data_sample_dict, self.config)
            all_summaries.append(summary)
            
            # Update comparison file incrementally (reused code from _save_comparison method)
            comparison_json = self.results_dir / "model_comparison.json"
            
            if comparison_json.exists():
                with open(comparison_json, 'r') as f:
                    existing_summaries = json.load(f)
                
                existing_result = next((s for s in existing_summaries if s.get('model') == model_name), None)
                
                if 'error' in summary and existing_result and 'error' not in existing_result:
                    logger.info(f"Keeping existing valid result for {model_name} (new run failed)")
                    all_summaries_to_save = existing_summaries
                else:
                    existing_summaries = [s for s in existing_summaries if s.get('model') != model_name]
                    existing_summaries.append(summary)
                    all_summaries_to_save = existing_summaries
            else:
                all_summaries_to_save = all_summaries
            
            with open(comparison_json, 'w') as f:
                json.dump(all_summaries_to_save, f, indent=2)
            logger.info(f"✓ Updated model comparison in {comparison_json}")
            
            logger.info(f"✓ Completed {model_name} benchmark")
            if self.interactive:
                input(f"You can now stop the {model_name} service. Press ENTER to continue...")
        
        logger.info(f"✓ Benchmark results saved to {comparison_json}")
        
        # Visualizations
        self._generate_visualizations(logger)
        
        logger.info("="*60)
        logger.info("SEQUENTIAL BENCHMARK COMPLETE!")
        logger.info("="*60)
    
    def _save_comparison(self, all_summaries: list, logger):
        """Save model comparison results."""
        comparison_json = self.results_dir / "model_comparison.json"
        
        # Load existing comparison if it exists
        if comparison_json.exists():
            with open(comparison_json, 'r') as f:
                existing_summaries = json.load(f)
            
            # For each new summary, check if we should keep old valid result
            summaries_to_add = []
            for new_summary in all_summaries:
                model_name = new_summary.get('model')
                existing_result = next((s for s in existing_summaries if s.get('model') == model_name), None)
                
                # Keep old valid result if new one failed
                if 'error' in new_summary and existing_result and 'error' not in existing_result:
                    logger.info(f"Keeping existing valid result for {model_name} (new run failed)")
                    summaries_to_add.append(existing_result)
                else:
                    summaries_to_add.append(new_summary)
            
            benchmarked_models = {s.get('model') for s in all_summaries}
            existing_summaries = [s for s in existing_summaries if s.get('model') not in benchmarked_models]
            all_summaries_to_save = existing_summaries + summaries_to_add
        else:
            all_summaries_to_save = all_summaries
        
        with open(comparison_json, 'w') as f:
            json.dump(all_summaries_to_save, f, indent=2)
        logger.info(f"✓ Model comparison saved to {comparison_json}")
    
    def _generate_visualizations(self, logger):
        """Generate visualization plots."""
        try:
            generate_all_plots(self.results_dir)
            logger.info("✓ Visualizations generated")
        except Exception as e:
            logger.error(f"✗ Visualizations failed: {e}")


def main():
    # Setup:
    parser = argparse.ArgumentParser(description='TTS Benchmark for Medical Speech')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("TTS BENCHMARK - Medical Conversational Speech")
    logger.info("="*60)
    
    logger.info(f"Loading config from {args.config}")
    benchmark = TTSBenchmark(args.config)
    
    # Load / create data sample:
    dataset_config = benchmark.config['dataset']
    sample_csv = benchmark.results_dir / "benchmark_sample.csv"
    
    if sample_csv.exists():
        logger.info(f"Loading existing sample from {sample_csv}")
        data_sample = pd.read_csv(sample_csv)
        if len(data_sample) != dataset_config['sample_size']:
            logger.warning(f"Existing sample size ({len(data_sample)}) does not match configured size ({dataset_config['sample_size']}). Creating new sample.")
            data_sample = create_benchmark_sample(
                metadata_path=dataset_config['metadata_path'],
                sample_size=dataset_config['sample_size'],
                output_path=str(sample_csv),
                strategy=dataset_config['sampling_strategy'],
                seed=dataset_config['random_seed']
            )
            logger.info(f"New sample created with {len(data_sample)} samples.")
    else:
        logger.info("Creating new benchmark sample...")
        data_sample = create_benchmark_sample(
            metadata_path=dataset_config['metadata_path'],
            sample_size=dataset_config['sample_size'],
            output_path=str(sample_csv),
            strategy=dataset_config['sampling_strategy'],
            seed=dataset_config['random_seed']
        )
    
    logger.info(f"Loaded {len(data_sample)} samples\n")
    
    # Run benchmark:
    benchmark.run(data_sample)


if __name__ == "__main__":
    main()
