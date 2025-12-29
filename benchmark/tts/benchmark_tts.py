"""
TTS Benchmark - Main Orchestrator

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)


class TTSBenchmark:
    """TTS Benchmark runner following WhisperBenchmark pattern."""
    
    def __init__(self, config_path='config.yaml'):
        """Load configuration and initialize paths."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['results_dir']).resolve()
        self.results_dir.mkdir(exist_ok=True)
        
        self.whisper_url = self.config['whisper_service']['url']
        self.nisqa_config = self.config.get('nisqa_service', None)
        self.tts_models = self.config['tts_models']
        self.save_audio = self.config.get('save_audio', True)
    
    def generate_audio(
        self,
        text: str,
        audio_path: Path,
        tts_client: TTSClient,
        asr_client: ASRClient,
        nisqa_client: NISQAClient = None
    ) -> dict:
        """
        Generate audio for a single text sample and evaluate it.
        
        Args:
            text: Text to synthesize
            audio_path: Where to save generated audio
            tts_client: TTS service client
            asr_client: ASR service client for round-trip evaluation
            nisqa_client: Optional NISQA client for MOS prediction
            
        Returns:
            Dictionary with all metrics for this sample
        """
        # Generate audio
        tts_result = tts_client.synthesize(text, str(audio_path))
        
        if not tts_result['success']:
            return {
                'text': text,
                'error': tts_result['error'],
                'success': False
            }
        
        # Transcribe audio (ASR round-trip)
        asr_result = asr_client.transcribe_file(str(audio_path))
        
        if not asr_result['success']:
            return {
                'text': text,
                'audio_path': str(audio_path),
                'generation_time': tts_result['generation_time'],
                'audio_duration': tts_result['duration'],
                'error': asr_result['error'],
                'success': False
            }
        
        # Calculate metrics
        transcription = asr_result['text']
        wer_score = calculate_wer(text, transcription)
        cer_score = calculate_cer(text, transcription)
        rtf = calculate_rtf(tts_result['generation_time'], tts_result['duration'])
        
        # Predict MOS quality score if NISQA is available
        mos_scores = {}
        if nisqa_client:
            nisqa_result = nisqa_client.predict_quality(str(audio_path))
            if nisqa_result['success']:
                mos_scores = {
                    'mos': nisqa_result['mos'],
                    'nisqa_latency': nisqa_result['latency']
                }
        
        # Return complete result
        result_entry = {
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
        return result_entry
    
    def benchmark_single_model(self, model_config: dict, data_sample: pd.DataFrame) -> dict:
        """
        Benchmark a single TTS model (like run_benchmark_for_model in WhisperBenchmark).
        
        Args:
            model_config: {'name': str, 'url': str}
            data_sample: DataFrame with 'text' and 'sample_id' columns
            
        Returns:
            Dictionary with aggregated results
        """
        model_name = model_config['name']
        model_url = model_config['url']
        
        logger.info(f"{'='*60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"{'='*60}")
        
        # Initialize clients
        tts_client = TTSClient(model_name, model_url)
        asr_client = ASRClient(self.whisper_url)
        
        # Initialize NISQA client if enabled
        nisqa_client = None
        if self.nisqa_config and self.nisqa_config.get('enabled', False):
            nisqa_client = NISQAClient(self.nisqa_config['url'])
        
        # Health checks
        logger.info("Running health checks...")
        if not tts_client.health_check():
            logger.error(f"✗ {model_name} service is not responding!")
            return {'error': f'{model_name} service not available'}
        
        if not asr_client.health_check():
            logger.error("✗ Whisper ASR service is not responding!")
            return {'error': 'Whisper service not available'}
        
        # Check NISQA if enabled
        if nisqa_client:
            if not nisqa_client.health_check():
                logger.warning("/!\ NISQA service not available - MOS scores will be skipped")
                nisqa_client = None
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_results_dir = self.results_dir / f"{timestamp}_{model_name}"
        audio_dir = model_results_dir / "audio_samples"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Results will be saved to: {model_results_dir}")
        
        # Metrics storage
        results = []
        
        # Process each sample
        logger.info(f"Processing {len(data_sample)} samples...")
        for idx, row in tqdm(data_sample.iterrows(), total=len(data_sample), desc=f"{model_name}"):
            sample_id = row['sample_id']
            text = row['text']
            
            # Generate audio and evaluate
            audio_path = audio_dir / f"sample_{sample_id:04d}.wav"
            result_entry = self.generate_audio(text, audio_path, tts_client, asr_client, nisqa_client)
            result_entry['sample_id'] = sample_id
            results.append(result_entry)
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_csv = model_results_dir / "detailed_results.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info(f"✓ Saved detailed results to {results_csv}")
        
        # Calculate summary statistics
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
            
            # Add MOS statistics if available
            if 'mos' in successful.columns:
                summary.update({
                    'mos_mean': float(successful['mos'].mean()),
                    'mos_median': float(successful['mos'].median()),
                    'mos_std': float(successful['mos'].std()),
                })
        
        # Save summary
        summary_json = model_results_dir / "summary.json"
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary to {summary_json}\n")
        
        # Print summary
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
        if not self.save_audio:
            import shutil
            shutil.rmtree(audio_dir)
            logger.info(f"✓ Cleaned up audio files (save_audio=False)")
        
        return summary
    
    def run(self, data_sample: pd.DataFrame):
        """
        Run benchmark for all configured TTS models.
        
        Args:
            data_sample: DataFrame with text samples to benchmark
        """
        all_summaries = []
        
        for model_config in self.tts_models:
            model_name = model_config['name']
            
            # Prompt user to start service
            logger.info(f"{'='*60}")
            logger.info(f"Ready to benchmark: {model_name}")
            logger.info(f"Service URL: {model_config['url']}")
            logger.info(f"{'='*60}")
            input(f"\n-->  Please ensure {model_name} service is running, then press ENTER to continue...\n")
            
            # Run benchmark for this model
            summary = self.benchmark_single_model(model_config, data_sample)
            all_summaries.append(summary)
            
            # Update model comparison file
            comparison_json = self.results_dir / "model_comparison.json"
            
            # Load existing comparison if it exists
            if comparison_json.exists():
                with open(comparison_json, 'r') as f:
                    existing_summaries = json.load(f)
                # Remove old results for this model if they exist
                existing_summaries = [s for s in existing_summaries if s.get('model') != model_name]
                # Add new results
                existing_summaries.append(summary)
                all_summaries_to_save = existing_summaries
            else:
                all_summaries_to_save = all_summaries
            
            # Save updated comparison
            with open(comparison_json, 'w') as f:
                json.dump(all_summaries_to_save, f, indent=2)
            logger.info(f"✓ Updated model comparison in {comparison_json}")
            
            # Prompt to stop service
            logger.info(f"✓ Completed {model_name} benchmark")
            input(f"You can now stop the {model_name} service. Press ENTER to continue...")
        
        # Final summary message
        logger.info(f"✓ Benchmark results saved to {comparison_json}")
        
        # Visualizations
        try:
            generate_all_plots(self.results_dir)
            logger.info("Visualizations generated")
        except Exception as e:
            logger.error(f"Visualizations failed: {e}")
        
        logger.info("="*60)
        logger.info("BENCHMARK COMPLETE!")
        logger.info("="*60)


def main():
    """Main entry point for TTS benchmark."""
    parser = argparse.ArgumentParser(description='TTS Benchmark for Medical Speech')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TTS BENCHMARK - Medical Conversational Speech")
    logger.info("="*60)
    
    # Initialize benchmark
    logger.info(f"Loading config from {args.config}")
    benchmark = TTSBenchmark(args.config)
    
    # Load or create data sample
    dataset_config = benchmark.config['dataset']
    sample_csv = benchmark.results_dir / "benchmark_sample.csv"
    
    # Load existing sample if available and verify size
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
    
    # Run benchmark for all models
    benchmark.run(data_sample)


if __name__ == "__main__":
    main()
