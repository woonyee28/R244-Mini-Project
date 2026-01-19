from codecarbon import OfflineEmissionsTracker
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from datasets import load_dataset
import os
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import torch

@dataclass
class VLLMConfig:
    block_size: int = 32
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    enable_profiling: bool = False
    disable_custom_all_reduce: bool = True
    dtype: Optional[str] = 'auto'
    seed: int = 42

    
    @property
    def total_gpus(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class EvaluationResult:
    model_loading_time: float
    throughput: float  # tokens/sec
    requests_per_sec: float
    total_time: float
    emissions: float  # kg CO2
    energy: float # kWh
    total_output_tokens: int
    num_requests: int
    energy_per_token: float # J
    config: VLLMConfig
    
    def to_dict(self) -> Dict:
        return asdict(self)


class VLLMEvaluator:    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        country_iso_code: str = "GBR",
    ):
        self.model_name = model_name
        self.country_iso_code = country_iso_code
        self.dataset = None
        self.prompts = None
    
    def load_dataset(self, dataset_name: str = "Open-Orca/OpenOrca", split: str = "train[:1000]"):
        print(f"Loading {dataset_name} dataset...")
        self.dataset = load_dataset(dataset_name, split=split)
        self.prompts = self.dataset["question"]
        print(f"Loaded {len(self.prompts)} prompts from dataset")
        return self.prompts
    
    def setup_gpus(self, gpu_ids: List[int], config: VLLMConfig) -> bool:
        if len(gpu_ids) != config.total_gpus:
            print(f"ERROR: Number of GPUs ({len(gpu_ids)}) doesn't match "
                  f"tensor_parallel_size * pipeline_parallel_size ({config.total_gpus})")
            return False
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if config.enable_profiling:
            os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
            os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1"
            os.environ["VLLM_TORCH_PROFILER_WITH_FLOPS"] = "1"

        return True
    
    def evaluate(
        self,
        config: VLLMConfig,
        gpu_ids: List[int],
        num_prompts: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        verbose: bool = True
    ) -> Optional[EvaluationResult]:
        """
        Evaluate vLLM with given configuration
        
        Args:
            config: VLLMConfig object
            gpu_ids: List of GPU IDs to use
            num_prompts: Number of prompts to process
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            verbose: Print detailed output
        
        Returns:
            EvaluationResult or None if evaluation fails
        """
        if not self.setup_gpus(gpu_ids, config):
            return None

        if self.prompts is None:
            self.load_dataset()
        
        sampled_prompts = self.prompts[:num_prompts] if num_prompts < len(self.prompts) else self.prompts
        
        if verbose:
            self._print_config(config, gpu_ids, num_prompts)
        
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=256)
            
        try:          
            with OfflineEmissionsTracker(
                country_iso_code=self.country_iso_code,
                log_level="error" if not verbose else "info"
            ) as tracker:
                
                if verbose:
                    print("Initializing vLLM...")

                start_time = time.time()
                llm = LLM(
                    model=self.model_name,
                    data_parallel_size=config.data_parallel_size,
                    tensor_parallel_size=config.tensor_parallel_size,
                    pipeline_parallel_size=config.pipeline_parallel_size,
                    block_size=config.block_size,
                    max_num_seqs=config.max_num_seqs,
                    disable_log_stats=not verbose,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    disable_custom_all_reduce=config.disable_custom_all_reduce,
                    enable_chunked_prefill=config.enable_chunked_prefill,
                    enable_prefix_caching=config.enable_prefix_caching,
                    dtype=config.dtype,
                    seed=config.seed,
                    enforce_eager=True,
                )

                inference_start_time = time.time()
                model_loading_time = time.time() - start_time
                
                if config.enable_profiling:
                    llm.start_profile()
                outputs = llm.generate(sampled_prompts, sampling_params)
                if config.enable_profiling:
                    llm.stop_profile()
                elapsed_time = time.time() - inference_start_time
                
            del llm
            torch.cuda.empty_cache()
            
            total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            throughput = total_output_tokens / elapsed_time
            requests_per_sec = len(outputs) / elapsed_time
            emissions = tracker.final_emissions
            energy = (
                (tracker.final_emissions_data.cpu_energy or 0) +
                (tracker.final_emissions_data.gpu_energy or 0) +
                (tracker.final_emissions_data.ram_energy or 0)
            )
            energy_per_token=energy*3600*1000 / total_output_tokens

            print(tracker.final_emissions_data)
            
            if verbose:
                print(f"\nResults:")
                print(f"  Model loading time: {model_loading_time:.2f}s")
                print(f"  Throughput: {throughput:.2f} tokens/sec")
                print(f"  Energy per token: {energy_per_token:.2f} J/token")
                print(f"  Requests/sec: {requests_per_sec:.2f}")
                print(f"  Total time: {elapsed_time:.2f}s")
                print(f"  Emissions: {emissions:.6f} kg CO2")
                print(f"  Total energy consumed: {energy:.6f} kWh")
            
            return EvaluationResult(
                model_loading_time=model_loading_time,
                throughput=throughput,
                requests_per_sec=requests_per_sec,
                total_time=elapsed_time,
                emissions=emissions,
                energy=energy,
                total_output_tokens=total_output_tokens,
                num_requests=len(outputs),
                energy_per_token=energy_per_token,
                config=config
            )
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_config(self, config: VLLMConfig, gpu_ids: List[int], num_prompts: int):
        print(f"\n{'='*70}")
        print(f"vLLM EVALUATION")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  GPUs: {gpu_ids}")
        print(f"  Tensor Parallel Size: {config.tensor_parallel_size}")
        print(f"  Pipeline Parallel Size: {config.pipeline_parallel_size}")
        print(f"  Total GPUs: {config.total_gpus}")
        print(f"  Block Size: {config.block_size}")
        print(f"  Batch Size: {config.max_num_seqs}")
        print(f"  Num Prompts: {num_prompts}")
        print(f"  Chunked Prefill: {config.enable_chunked_prefill}")
        print(f"  Prefix Caching: {config.enable_prefix_caching}")
        print(f"{'='*70}\n")


def main():
    """Original main function for standalone evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run vLLM with multi-GPU profiling and CodeCarbon')
    parser.add_argument('--block_size', type=int, default=16, help='Block size for Paged Attention')
    parser.add_argument('--max_num_seqs', type=int, default=1024, help='Batch size (max_num_seqs)')
    parser.add_argument('--max_num_batched_tokens', type=int, default=8192, help='Max num batched tokens')
    parser.add_argument('--num_prompts', type=int, default=1000, help='Number of prompts to process')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Number of GPUs for tensor parallelism (default: 1)')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1,
                       help='Number of GPUs for pipeline parallelism (default: 1)')
    parser.add_argument('--data_parallel_size', type=int, default=1,
                       help='Number of GPUs for data parallelism (default: 1)')
    parser.add_argument('--gpus', type=str, default='4',
                       help='Comma-separated GPU IDs to use (e.g., "4,5" for GPUs 4 and 5)')
    parser.add_argument('--enable_chunked_prefill', action='store_true', default=False)
    parser.add_argument('--enable_prefix_caching', action='store_true', default=False)
    parser.add_argument('--dtype', type=str, default='auto')
    parser.add_argument('--enable_profiling', action='store_true', default=False)

    args = parser.parse_args()
    
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    config = VLLMConfig(
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_profiling=args.enable_profiling,
        dtype=args.dtype
        )
    
    evaluator = VLLMEvaluator()
    evaluator.load_dataset()
    
    result = evaluator.evaluate(
        config=config,
        gpu_ids=gpu_ids,
        num_prompts=args.num_prompts,
        verbose=True
    )
    
    if result:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\nResults saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
