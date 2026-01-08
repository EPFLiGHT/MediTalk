import torch
import soundfile as sf
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
import os
import logging
import re
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import threading
import gc
import time
from typing import List, Tuple, Optional
from tqdm import tqdm
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Set PyTorch CUDA memory allocation config to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================================
# MULTIPROCESSING WORKER FUNCTION (must be at module level to be picklable)
# ============================================================================

def _worker_process_chunk(chunk_idx: int, chunk: str, voice: str, device: str, model_name: str, hf_token: str) -> Tuple[int, Optional[np.ndarray]]:
    """
    Worker function for multiprocessing. Each process loads its own model instance.
    This function runs in a separate Python process with its own memory space.
    
    Args:
        chunk_idx: Index of this chunk
        chunk: Text to synthesize
        voice: Voice name
        device: CUDA device (e.g., "cuda:0")
        model_name: HuggingFace model name
        hf_token: HuggingFace token for authentication
        
    Returns:
        Tuple of (chunk_idx, audio_array) or (chunk_idx, None) on failure
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    import logging
    
    # Set up logging for this process
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        start_time = time.time()
        logger.info(f"[Process {os.getpid()}] Chunk {chunk_idx+1}: Loading model on {device}...")
        
        # Set CUDA device for this process
        torch.cuda.set_device(device)
        
        # Load Orpheus model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},  # Force to specific GPU
            token=hf_token
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        
        # Load SNAC model
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        
        load_time = time.time() - start_time
        logger.info(f"[Process {os.getpid()}] Chunk {chunk_idx+1}: Models loaded in {load_time:.2f}s")
        
        # Generate audio
        gen_start = time.time()
        logger.info(f"[Process {os.getpid()}] Chunk {chunk_idx+1}: Generating (len={len(chunk)} chars)...")
        
        # Prepare input
        prompt = f"{voice}: {chunk}"
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        modified_input = torch.cat([start_token, input_ids, end_tokens], dim=1)
        input_ids = modified_input.to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10000,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=128258,
            )

        # Post-process tokens
        token_to_find = 128257
        token_to_remove = 128258
        indices = (generated == token_to_find).nonzero(as_tuple=True)
        cropped = generated[:, indices[1][-1]+1:] if len(indices[1]) > 0 else generated
        cropped = cropped[cropped != token_to_remove]

        if len(cropped) == 0:
            logger.warning(f"[Process {os.getpid()}] Chunk {chunk_idx+1}: No valid audio tokens")
            return (chunk_idx, None)

        code_list = [(t - 128266).item() for t in cropped]

        # Decode with SNAC
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
        
        codes = [
            torch.tensor(layer_1).unsqueeze(0).to(device),
            torch.tensor(layer_2).unsqueeze(0).to(device),
            torch.tensor(layer_3).unsqueeze(0).to(device)
        ]
        
        with torch.no_grad():
            audio = snac_model.decode(codes)
        
        audio_numpy = audio.detach().squeeze().to("cpu").numpy()
        
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        
        logger.info(f"[Process {os.getpid()}] Chunk {chunk_idx+1}: Complete! Gen={gen_time:.2f}s, Total={total_time:.2f}s")
        
        # Cleanup
        del model, tokenizer, snac_model, generated, audio
        torch.cuda.empty_cache()
        
        return (chunk_idx, audio_numpy)
        
    except Exception as e:
        logger.error(f"[Process {os.getpid()}] Chunk {chunk_idx+1}: Error - {e}")
        import traceback
        traceback.print_exc()
        return (chunk_idx, None)

# ============================================================================
# END WORKER FUNCTION
# ============================================================================

class OrpheusTTS:
    def __init__(self, model_name="canopylabs/orpheus-3b-0.1-ft", device=None, fallback_model=None, max_parallel_chunks=16, use_multi_gpu=False):
        logger.info("Initializing OrpheusTTS...")
        
        # For now, disable multi-GPU due to model sharding conflicts
        self.use_multi_gpu = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.devices = [self.device]
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {num_gpus} GPUs - model will use device_map='auto' for layer distribution")
        
        self.fallback_model = fallback_model
        
        # Parallel processing configuration
        self.max_parallel_chunks = max_parallel_chunks
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_chunks)
        
        # Thread lock to serialize GPU model.generate() calls
        # PyTorch models with device_map="auto" are not thread-safe
        self.gpu_lock = threading.Lock()
        
        logger.info(f"Parallel chunk processing enabled: max {max_parallel_chunks} concurrent chunks")
        logger.info("Note: GPU operations are serialized with a lock for thread safety")

        # Handle Hugging Face authentication
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            logger.info("Authenticating with Hugging Face...")
            try:
                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Hugging Face: {e}")
        else:
            logger.warning("No HUGGINGFACE_TOKEN found. You may not be able to access gated models.")
            logger.info("To use Orpheus, you need to:")
            logger.info("   1. Get a Hugging Face token: https://huggingface.co/settings/tokens")
            logger.info("   2. Request access to: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft")
            logger.info("   3. Set HUGGINGFACE_TOKEN environment variable")

        # Try to load the model with error handling
        models_to_try = [model_name]
        if fallback_model:
            models_to_try.append(fallback_model)

        for model_to_load in models_to_try:
            try:
                logger.info(f"Attempting to load model: {model_to_load}")
                
                # Load SNAC decoder on primary device
                logger.info("Loading SNAC decoder...")
                primary_device = self.device
                self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(primary_device)
                logger.info("SNAC decoder loaded successfully")

                # Download model weights/config
                logger.info("Downloading Orpheus model from Hugging Face...")
                snapshot_download(
                    repo_id=model_to_load,
                    allow_patterns=["config.json", "*.safetensors", "model.safetensors.index.json"],
                    token=hf_token  # Pass token explicitly
                )

                logger.info("Loading Orpheus model...")
                
                # Load model with device_map="auto" to distribute layers across all GPUs
                # This provides automatic parallelism at the layer level
                logger.info("Loading model with automatic device mapping for layer distribution...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",  # Automatically distribute layers across all available GPUs
                    trust_remote_code=True,
                    token=hf_token
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_to_load, token=hf_token)
                
                logger.info(f"Successfully loaded model: {model_to_load}")
                self.current_model = model_to_load
                return
                
            except Exception as e:
                logger.error(f"Failed to load model {model_to_load}: {e}")
                if model_to_load == models_to_try[-1]:  # Last model in list
                    logger.error("All model loading attempts failed!")
                    raise e
                else:
                    logger.info(f"Trying next model...")
                    continue
    
    def _get_gpu_memory_info(self) -> List[Tuple[int, float, float]]:
        """
        Get memory information for all available GPUs.
        Returns: List of (gpu_id, free_memory_gb, total_memory_gb)
        """
        if not torch.cuda.is_available():
            return []
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            free = total - reserved
            gpu_info.append((i, free, total))
            logger.debug(f"GPU {i}: {free:.2f}GB free / {total:.2f}GB total (allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB)")
        
        return gpu_info
    
    def _calculate_optimal_instances(self, num_chunks: int) -> Tuple[int, List[int]]:
        """
        Calculate optimal number of model instances and their GPU assignment.
        
        Args:
            num_chunks: Number of text chunks to generate
            
        Returns:
            (num_instances, gpu_assignments) where gpu_assignments[i] is the GPU ID for instance i
        """
        # Constants
        MODEL_SIZE_GB = 4.5  # Orpheus model size including overhead
        MIN_FREE_GB = 10  # Minimum free memory to keep per GPU
        MAX_INSTANCES_PER_GPU = 4  # Safety limit
        
        gpu_info = self._get_gpu_memory_info()
        if not gpu_info:
            logger.warning("No GPUs available, using CPU (sequential only)")
            return 1, [None]
        
        # Calculate how many instances each GPU can handle
        gpu_capacity = []
        for gpu_id, free_gb, total_gb in gpu_info:
            available = free_gb - MIN_FREE_GB
            if available > MODEL_SIZE_GB:
                max_instances = min(int(available / MODEL_SIZE_GB), MAX_INSTANCES_PER_GPU)
                gpu_capacity.append((gpu_id, max_instances))
                logger.info(f"GPU {gpu_id}: Can fit {max_instances} instances ({free_gb:.1f}GB free)")
            else:
                logger.warning(f"GPU {gpu_id}: Insufficient memory ({free_gb:.1f}GB free, need {MODEL_SIZE_GB + MIN_FREE_GB:.1f}GB)")
        
        if not gpu_capacity:
            logger.warning("No GPUs have sufficient memory, falling back to sequential")
            return 1, [0]  # Use GPU 0 with sequential processing
        
        # Calculate total instances we can create
        total_capacity = sum(cap for _, cap in gpu_capacity)
        num_instances = min(num_chunks, total_capacity)
        
        # Distribute instances across GPUs (round-robin)
        gpu_assignments = []
        gpu_idx = 0
        for i in range(num_instances):
            gpu_id, _ = gpu_capacity[gpu_idx % len(gpu_capacity)]
            gpu_assignments.append(gpu_id)
            gpu_idx += 1
        
        logger.info(f"Optimal configuration: {num_instances} instances for {num_chunks} chunks")
        logger.info(f"GPU distribution: {dict((f'cuda:{g}', gpu_assignments.count(g)) for g in set(gpu_assignments))}")
        
        return num_instances, gpu_assignments

    def synthesize(self, text, voice="tara", output_path="output.wav", generate_in_parallel=True):
        """
        Generate TTS audio from text and save to output_path.
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            output_path: Path to save audio file
            generate_in_parallel: Whether to use dynamic multi-instance parallel processing (default: True)
        
        Returns:
            output_path: Path to generated audio file
        """
        start_time = time.time()
        timings = {}
        
        try:
            # Step 1: Clear GPU cache
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timings['gpu_cache_clear'] = time.time() - t0
            
            # Step 2: Split text into chunks
            t0 = time.time()
            logger.info(f"Synthesizing text (length={len(text)} chars) with voice '{voice}'")
            chunks = self._chunk_text(text, max_chars=200)
            num_chunks = len(chunks)
            logger.info(f"Text split into {num_chunks} chunks")
            timings['text_chunking'] = time.time() - t0
            
            # Step 3: Choose processing mode
            if generate_in_parallel and num_chunks > 1:
                logger.info("=== PARALLEL MODE: Dynamic multi-instance processing ===")
                audio_pieces = self._synthesize_with_dynamic_instances(chunks, voice, timings)
            else:
                logger.info("=== SEQUENTIAL MODE: Single instance processing ===")
                audio_pieces = self._synthesize_sequential(chunks, voice, timings)
            
            if len(audio_pieces) == 0:
                raise ValueError("No audio was generated for any chunk")

            # Step 4: Concatenate audio pieces
            t0 = time.time()
            logger.info("Concatenating audio pieces...")
            final_audio = self._concatenate_audio(audio_pieces, sample_rate=24000)
            timings['concatenation'] = time.time() - t0

            # Step 5: Save to file
            t0 = time.time()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, final_audio, 24000)
            timings['file_write'] = time.time() - t0
            
            # Total time
            total_time = time.time() - start_time
            timings['total'] = total_time
            
            # Log performance summary
            logger.info("=" * 60)
            logger.info("PERFORMANCE SUMMARY")
            logger.info("=" * 60)
            mode = 'SEQUENTIAL'
            if generate_in_parallel and num_chunks > 1:
                mode = 'MULTIPROCESSING (separate processes)'
            logger.info(f"Mode: {mode}")
            logger.info(f"Total chunks: {num_chunks}")
            logger.info(f"Text chunking: {timings['text_chunking']:.2f}s")
            if 'memory_check' in timings:
                logger.info(f"Memory check: {timings['memory_check']:.2f}s")
            if 'instance_loading' in timings:
                logger.info(f"Model loading (per process): {timings['instance_loading']:.2f}s")
            if 'parallel_generation' in timings:
                logger.info(f"Parallel generation (wall time): {timings['parallel_generation']:.2f}s")
            if 'sequential_generation' in timings:
                logger.info(f"Sequential generation: {timings['sequential_generation']:.2f}s")
            if 'instance_cleanup' in timings:
                logger.info(f"Instance cleanup: {timings['instance_cleanup']:.2f}s")
            logger.info(f"Audio concatenation: {timings['concatenation']:.2f}s")
            logger.info(f"File write: {timings['file_write']:.2f}s")
            logger.info(f"TOTAL TIME: {total_time:.2f}s")
            logger.info(f"Average per chunk: {total_time/num_chunks:.2f}s")
            logger.info("=" * 60)
            
            logger.info(f"Audio saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise e
    
    def _synthesize_with_dynamic_instances(self, chunks: List[str], voice: str, timings: dict) -> List[np.ndarray]:
        """
        Synthesize chunks using multiprocessing with separate processes.
        
        This method:
        1. Checks available GPU memory
        2. Spawns N worker processes (one per chunk, distributed across GPUs)
        3. Each process loads its own model instance and generates audio
        4. Collects results from all processes
        
        This provides TRUE parallelism by bypassing Python's GIL.
        
        Args:
            chunks: List of text chunks to synthesize
            voice: Voice to use
            timings: Dictionary to store timing information
            
        Returns:
            List of audio numpy arrays in order
        """
        num_chunks = len(chunks)
        
        # Step 1: Calculate optimal instances
        t0 = time.time()
        num_instances, gpu_assignments = self._calculate_optimal_instances(num_chunks)
        timings['memory_check'] = time.time() - t0
        timings['num_instances'] = num_instances
        
        if num_instances == 1:
            logger.warning("Insufficient GPU memory for parallel instances, falling back to sequential")
            return self._synthesize_sequential(chunks, voice, timings)
        
        # Step 2: Prepare worker arguments
        # Each worker will load its own model in its own process
        logger.info(f"=== MULTIPROCESSING MODE: Spawning {num_chunks} worker processes ===")
        logger.info(f"GPU distribution: {dict((f'cuda:{g}', gpu_assignments.count(g)) for g in set(gpu_assignments))}")
        
        # Prepare arguments for each worker
        worker_args = []
        for i, chunk in enumerate(chunks):
            gpu_id = gpu_assignments[i % len(gpu_assignments)]  # Round-robin
            device = f"cuda:{gpu_id}"
            worker_args.append((
                i,                              # chunk_idx
                chunk,                          # chunk text
                voice,                          # voice
                device,                         # device
                self.current_model,             # model_name
                os.getenv('HUGGINGFACE_TOKEN')  # hf_token
            ))
        
        # Step 3: Spawn processes and run in parallel
        t0 = time.time()
        
        print("\n" + "="*80)
        print(f"SPAWNING {num_chunks} WORKER PROCESSES")
        print("="*80)
        
        try:
            # Use spawn method to ensure clean process initialization (especially important for CUDA)
            ctx = mp.get_context('spawn')
            
            with ctx.Pool(processes=num_chunks) as pool:
                # Run all workers in parallel
                logger.info(f"Starting parallel generation across {num_chunks} processes...")
                results = pool.starmap(_worker_process_chunk, worker_args)
            
            timings['parallel_generation'] = time.time() - t0
            
            print("="*80)
            print(f"ALL PROCESSES COMPLETE in {timings['parallel_generation']:.2f}s")
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Falling back to sequential mode")
            return self._synthesize_sequential(chunks, voice, timings)
        
        # Step 4: Process results (sort by chunk index and extract audio)
        results.sort(key=lambda x: x[0])  # Sort by chunk_idx
        audio_pieces = [audio for idx, audio in results if audio is not None]
        
        success_count = len(audio_pieces)
        logger.info(f"Parallel generation complete: {success_count}/{num_chunks} chunks successful")
        
        if success_count < num_chunks:
            logger.warning(f"Some chunks failed: {num_chunks - success_count} failures")
        
        # No cleanup needed - each process handles its own cleanup
        timings['instance_loading'] = 0  # Handled inside worker processes
        timings['instance_cleanup'] = 0  # Handled inside worker processes
        
        return audio_pieces
    
    def _create_lightweight_instance(self, device: str):
        """
        Create a lightweight OrpheusTTS instance on a specific device.
        Returns a dict with model, tokenizer, and snac_model.
        """
        instance = {}
        
        # Load tokenizer (shared, minimal memory)
        instance['tokenizer'] = self.tokenizer
        
        # Load SNAC decoder on the device
        instance['snac_model'] = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        
        # Load Orpheus model on specific device (NOT device_map="auto")
        instance['model'] = AutoModelForCausalLM.from_pretrained(
            self.current_model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},  # Force to specific device
            trust_remote_code=True,
            token=os.getenv('HUGGINGFACE_TOKEN')
        )
        
        instance['device'] = device
        
        return instance
    
    def _cleanup_instance(self, instance: dict):
        """Clean up a model instance to free GPU memory."""
        try:
            if 'model' in instance:
                del instance['model']
            if 'snac_model' in instance:
                del instance['snac_model']
            
            # Clear CUDA cache for this device
            if 'device' in instance and instance['device'].startswith('cuda'):
                device_id = int(instance['device'].split(':')[1])
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during instance cleanup: {e}")
    
    def _parallel_generate_with_instances(self, chunks: List[str], voice: str, instances: List[Tuple]) -> List[np.ndarray]:
        """
        Generate audio for all chunks in parallel using multiple model instances.
        
        Args:
            chunks: List of text chunks
            voice: Voice to use
            instances: List of (index, instance_dict, device) tuples
            
        Returns:
            List of audio arrays in original chunk order
        """
        num_chunks = len(chunks)
        num_instances = len(instances)
        
        # Assign chunks to instances
        chunk_assignments = []
        for i, chunk in enumerate(chunks):
            instance_idx = i % num_instances  # Round-robin assignment
            _, instance, device = instances[instance_idx]
            chunk_assignments.append((i, chunk, instance, device))
        
        # Use ThreadPoolExecutor for parallel execution with nested progress bars
        results = [None] * num_chunks
        
        logger.info(f"Generating {num_chunks} chunks in parallel...")
        print("\n" + "="*80)
        print("PARALLEL GENERATION STARTED")
        print("="*80)
        
        # Create progress bars for each chunk (position 1 to num_chunks)
        chunk_pbars = {}
        for i in range(num_chunks):
            chunk_pbars[i] = tqdm(
                total=100,
                desc=f"  Chunk {i+1:2d}/{num_chunks}",
                position=i+1,
                leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar:20}| [{elapsed}<{remaining}]',
                ncols=80
            )
        
        # Create overall progress bar (position 0, bigger)
        overall_pbar = tqdm(
            total=num_chunks,
            desc="OVERALL PROGRESS",
            position=0,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar:40}| {n}/{total} chunks [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=100
        )
        
        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            # Submit all tasks
            future_to_idx = {}
            for chunk_idx, chunk, instance, device in chunk_assignments:
                future = executor.submit(
                    self._generate_single_chunk_with_instance,
                    chunk_idx, chunk, voice, instance, device, chunk_pbars[chunk_idx]
                )
                future_to_idx[future] = chunk_idx
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                chunk_idx = future_to_idx[future]
                try:
                    audio = future.result()
                    if audio is not None:
                        results[chunk_idx] = audio
                        chunk_pbars[chunk_idx].n = 100
                        chunk_pbars[chunk_idx].refresh()
                    else:
                        logger.warning(f"[Chunk {chunk_idx+1}] Failed to generate")
                        chunk_pbars[chunk_idx].set_description(f"  Chunk {chunk_idx+1:2d}/{num_chunks} [FAILED]")
                except Exception as e:
                    logger.error(f"[Chunk {chunk_idx+1}] Error: {e}")
                    chunk_pbars[chunk_idx].set_description(f"  Chunk {chunk_idx+1:2d}/{num_chunks} [ERROR]")
                
                overall_pbar.update(1)
        
        # Close all progress bars
        overall_pbar.close()
        for pbar in chunk_pbars.values():
            pbar.close()
        
        print("="*80)
        print("PARALLEL GENERATION COMPLETE")
        print("="*80 + "\n")
        
        # Filter out None results
        audio_pieces = [r for r in results if r is not None]
        
        logger.info(f"Parallel generation complete: {len(audio_pieces)}/{num_chunks} chunks successful")
        
        return audio_pieces
    
    def _generate_single_chunk_with_instance(self, chunk_idx: int, chunk: str, voice: str, instance: dict, device: str, pbar: tqdm = None) -> Optional[np.ndarray]:
        """
        Generate audio for a single chunk using a specific model instance.
        This runs in a thread pool worker.
        
        Args:
            chunk_idx: Index of the chunk
            chunk: Text to synthesize
            voice: Voice to use
            instance: Model instance dict
            device: Device name
            pbar: Optional progress bar for this chunk
        """
        try:
            if pbar:
                pbar.set_description(f"  Chunk {chunk_idx+1:2d} Preparing...")
                pbar.update(5)
            
            logger.info(f"[Chunk {chunk_idx+1}] Generating on {device} (len={len(chunk)} chars)")
            
            model = instance['model']
            tokenizer = instance['tokenizer']
            snac_model = instance['snac_model']
            
            # Prepare input
            if pbar:
                pbar.set_description(f"  Chunk {chunk_idx+1:2d} Tokenizing...")
                pbar.update(5)
            
            prompt = f"{voice}: {chunk}"
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            modified_input = torch.cat([start_token, input_ids, end_tokens], dim=1)

            input_ids = modified_input.to(device)
            attention_mask = torch.ones_like(input_ids)

            # Generate
            if pbar:
                pbar.set_description(f"  Chunk {chunk_idx+1:2d} Generating...")
                pbar.update(10)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10000,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    eos_token_id=128258,
                )
            
            if pbar:
                pbar.set_description(f"  Chunk {chunk_idx+1:2d} Post-processing...")
                pbar.update(50)  # Generation is the longest step

            # Post-process tokens
            token_to_find = 128257
            token_to_remove = 128258
            indices = (generated == token_to_find).nonzero(as_tuple=True)
            cropped = generated[:, indices[1][-1]+1:] if len(indices[1]) > 0 else generated
            cropped = cropped[cropped != token_to_remove]

            if len(cropped) == 0:
                logger.warning(f"[Chunk {chunk_idx+1}] No valid audio tokens generated")
                if pbar:
                    pbar.set_description(f"  Chunk {chunk_idx+1:2d} No tokens")
                return None

            code_list = [(t - 128266).item() for t in cropped]

            # Decode with instance's SNAC
            if pbar:
                pbar.set_description(f"  Chunk {chunk_idx+1:2d} Decoding...")
                pbar.update(20)
            
            audio = self._decode_snac_with_instance(code_list, snac_model, device)
            audio_numpy = audio.detach().squeeze().to("cpu").numpy()
            
            if pbar:
                pbar.set_description(f"  Chunk {chunk_idx+1:2d} Complete")
                pbar.update(10)  # Finish to 100%
            
            return audio_numpy
            
        except Exception as e:
            logger.error(f"[Chunk {chunk_idx+1}] Error during generation: {e}")
            if pbar:
                pbar.set_description(f"  Chunk {chunk_idx+1:2d} ERROR")
            return None
    
    def _decode_snac_with_instance(self, code_list, snac_model, device):
        """Decode audio using a specific SNAC instance."""
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
        
        codes = [torch.tensor(layer_1).unsqueeze(0).to(device),
                torch.tensor(layer_2).unsqueeze(0).to(device),
                torch.tensor(layer_3).unsqueeze(0).to(device)]
        audio_hat = snac_model.decode(codes)
        return audio_hat
    
    async def _synthesize_parallel(self, chunks, voice):
        """
        [DEPRECATED] Old async parallel method with lock.
        Use _synthesize_with_dynamic_instances instead.
        """
        logger.info(f"Starting parallel synthesis with max {self.max_parallel_chunks} concurrent chunks")
        
        # Semaphore to limit concurrent GPU operations
        semaphore = asyncio.Semaphore(self.max_parallel_chunks)
        
        # Create tasks for all chunks with their index
        tasks = []
        for i, chunk in enumerate(chunks):
            task = self._synthesize_chunk_async(i, chunk, voice, semaphore)
            tasks.append(task)
        
        # Wait for all chunks to complete (order is preserved by gather)
        indexed_results = await asyncio.gather(*tasks)
        
        # Sort by index to ensure correct order (though gather preserves order)
        indexed_results.sort(key=lambda x: x[0])
        
        # Extract audio pieces in order
        audio_pieces = [audio for _, audio in indexed_results if audio is not None]
        
        logger.info(f"Parallel synthesis complete: {len(audio_pieces)}/{len(chunks)} chunks successful")
        return audio_pieces
    
    async def _synthesize_chunk_async(self, index, chunk, voice, semaphore):
        """
        Asynchronously synthesize a single chunk with GPU concurrency control.
        Returns (index, audio_numpy) to preserve order.
        """
        async with semaphore:  # Limit concurrent GPU operations
            logger.info(f"[Chunk {index+1}] Starting synthesis (len={len(chunk)} chars)")
            
            # Run GPU-intensive work in executor to avoid blocking
            loop = asyncio.get_event_loop()
            audio_numpy = await loop.run_in_executor(
                self.executor,
                partial(self._synthesize_single_chunk, chunk, voice, index)
            )
            
            if audio_numpy is not None:
                logger.info(f"[Chunk {index+1}] Synthesis complete")
            else:
                logger.warning(f"[Chunk {index+1}] Synthesis failed")
            
            return (index, audio_numpy)
    
    def _synthesize_single_chunk(self, chunk, voice, index=0):
        """
        Synthesize a single chunk of text. This is the GPU-intensive operation.
        Returns audio_numpy or None if failed.
        
        Note: With device_map="auto", the model handles GPU placement internally.
        We just send inputs to the primary device and the model routes them.
        Uses a lock to serialize GPU model.generate() calls for thread safety.
        """
        try:
            prompt = f"{voice}: {chunk}"
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            modified_input = torch.cat([start_token, input_ids, end_tokens], dim=1)

            # Send to primary device - model's device_map handles internal routing
            input_ids = modified_input.to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Lock around model.generate() for thread safety with device_map="auto"
            with self.gpu_lock:
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=10000,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        eos_token_id=128258,
                    )

            # Post-process tokens (can be parallel, outside the lock)
            token_to_find = 128257
            token_to_remove = 128258
            indices = (generated == token_to_find).nonzero(as_tuple=True)
            cropped = generated[:, indices[1][-1]+1:] if len(indices[1]) > 0 else generated
            cropped = cropped[cropped != token_to_remove]

            if len(cropped) == 0:
                logger.warning(f"[Chunk {index+1}] No valid audio tokens generated")
                return None

            code_list = [(t - 128266).item() for t in cropped]

            # Decode to audio for this chunk (SNAC is on primary device)
            audio = self._decode_snac(code_list)
            audio_numpy = audio.detach().squeeze().to("cpu").numpy()
            
            return audio_numpy
            
        except Exception as e:
            logger.error(f"[Chunk {index+1}] Error during synthesis: {e}")
            return None
    
    def _synthesize_sequential(self, chunks, voice, timings):
        """
        Sequential synthesis using the main model instance.
        Includes per-chunk timing.
        """
        logger.info("Using sequential synthesis")
        audio_pieces = []
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        
        t_total = time.time()
        chunk_times = []
        
        print("\n" + "="*80)
        print("SEQUENTIAL GENERATION STARTED")
        print("="*80)
        
        # Use tqdm for progress bar with nicer formatting
        pbar = tqdm(
            enumerate(chunks), 
            total=len(chunks), 
            desc="SEQUENTIAL PROGRESS",
            bar_format='{desc}: {percentage:3.0f}%|{bar:40}| {n}/{total} chunks [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=100
        )
        
        for i, chunk in pbar:
            t_chunk = time.time()
            pbar.set_description(f"SEQUENTIAL PROGRESS - Chunk {i+1}/{len(chunks)}")
            logger.info(f"[Chunk {i+1}/{len(chunks)}] Synthesizing (len={len(chunk)} chars)")
            prompt = f"{voice}: {chunk}"

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            modified_input = torch.cat([start_token, input_ids, end_tokens], dim=1)

            input_ids = modified_input.to(self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10000,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    eos_token_id=128258,
                )

            # Post-process tokens
            token_to_find = 128257
            token_to_remove = 128258
            indices = (generated == token_to_find).nonzero(as_tuple=True)
            cropped = generated[:, indices[1][-1]+1:] if len(indices[1]) > 0 else generated
            cropped = cropped[cropped != token_to_remove]

            if len(cropped) == 0:
                logger.warning(f"[Chunk {i+1}] No valid audio tokens generated; skipping")
                continue

            code_list = [(t - 128266).item() for t in cropped]

            # Decode to audio for this chunk
            audio = self._decode_snac(code_list)
            audio_numpy = audio.detach().squeeze().to("cpu").numpy()
            audio_pieces.append(audio_numpy)
            
            chunk_time = time.time() - t_chunk
            chunk_times.append(chunk_time)
            logger.info(f"[Chunk {i+1}/{len(chunks)}] Complete in {chunk_time:.2f}s")
        
        pbar.close()
        
        print("="*80)
        print("SEQUENTIAL GENERATION COMPLETE")
        print("="*80 + "\n")
        
        timings['sequential_generation'] = time.time() - t_total
        timings['avg_chunk_time'] = sum(chunk_times) / len(chunk_times) if chunk_times else 0
        timings['chunk_times'] = chunk_times
        
        return audio_pieces

    def _decode_snac(self, code_list):
        """Official redistribution logic from orpheus_0_1_finetune_inference.py"""
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
        
        # SNAC decoder is on primary device
        codes = [torch.tensor(layer_1).unsqueeze(0).to(self.device),
                torch.tensor(layer_2).unsqueeze(0).to(self.device),
                torch.tensor(layer_3).unsqueeze(0).to(self.device)]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat

    def _chunk_text(self, text, max_chars=150):
        """Split text into sentence-aware chunks not exceeding max_chars.

        Tries to split on sentence boundaries (., ?, !) and falls back to whitespace splitting.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return [text]

        sentences = re.split(r'(?<=[\.\?!])\s+', text)
        chunks = []
        current = []
        current_len = 0
        for s in sentences:
            s_len = len(s)
            if current_len + s_len + (1 if current_len>0 else 0) <= max_chars:
                current.append(s)
                current_len += s_len + (1 if current_len>0 else 0)
            else:
                if current:
                    chunks.append(' '.join(current))
                # If single sentence is longer than max_chars, we must split it
                if s_len > max_chars:
                    # fallback: split by words
                    words = s.split(' ')
                    part = []
                    part_len = 0
                    for w in words:
                        if part_len + len(w) + (1 if part_len>0 else 0) <= max_chars:
                            part.append(w)
                            part_len += len(w) + (1 if part_len>0 else 0)
                        else:
                            chunks.append(' '.join(part))
                            part = [w]
                            part_len = len(w)
                    if part:
                        chunks.append(' '.join(part))
                    current = []
                    current_len = 0
                else:
                    current = [s]
                    current_len = s_len

        if current:
            chunks.append(' '.join(current))
        return chunks

    def _concatenate_audio(self, pieces, sample_rate=24000):
        """Concatenate numpy audio arrays and apply short crossfade between pieces to avoid clicks."""
        if len(pieces) == 1:
            return pieces[0]

        # Determine fade length (in samples) - 10ms
        fade_ms = 10
        fade_len = int(sample_rate * (fade_ms/1000.0))

        out = pieces[0].copy()
        for p in pieces[1:]:
            # apply crossfade
            if fade_len > 0 and len(out) >= fade_len and len(p) >= fade_len:
                tail = out[-fade_len:]
                head = p[:fade_len]
                # linear fade
                fade_out = np.linspace(1.0, 0.0, fade_len)
                fade_in = np.linspace(0.0, 1.0, fade_len)
                cross = tail * fade_out + head * fade_in
                out = np.concatenate([out[:-fade_len], cross, p[fade_len:]])
            else:
                out = np.concatenate([out, p])
        return out