from codecarbon import OfflineEmissionsTracker
from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
import argparse
from datetime import datetime
from pathlib import Path
import time

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1"
os.environ["VLLM_TORCH_PROFILER_WITH_FLOPS"] = "1"

print("Loading OpenOrca dataset...")
dataset = load_dataset("Open-Orca/OpenOrca", split="train[:1000]")
prompts = dataset["question"]
print(f"Loaded {len(prompts)} prompts from OpenOrca dataset")

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    parser = argparse.ArgumentParser(description='Run vLLM with multi-GPU profiling and CodeCarbon')
    parser.add_argument('--block_size', type=int, default=32, help='Block size for Paged Attention')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (max_num_seqs)')
    parser.add_argument('--num_prompts', type=int, default=1000, help='Number of prompts to process')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Number of GPUs for tensor parallelism (default: 1)')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1,
                       help='Number of GPUs for pipeline parallelism (default: 1)')
    parser.add_argument('--gpus', type=str, default='4',
                       help='Comma-separated GPU IDs to use (e.g., "4,5" for GPUs 4 and 5)')
    parser.add_argument('enable_chunked_prefill', type=bool, default=True)
    parser.add_argument('enable_prefix_caching', type=bool, default=False)

    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size

    if len(gpu_ids) != total_gpus:
        print(f"ERROR: Number of GPUs ({len(gpu_ids)}) doesn't match "
              f"tensor_parallel_size * pipeline_parallel_size ({total_gpus})")
        print(f"Expected {total_gpus} GPUs, got {len(gpu_ids)}: {gpu_ids}")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"vLLM MULTI-GPU PROFILING")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Model: mistralai/Mistral-7B-Instruct-v0.1")
    print(f"  GPUs: {gpu_ids}")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"  Pipeline Parallel Size: {args.pipeline_parallel_size}")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Block Size: {args.block_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Num Prompts: {args.num_prompts}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Power Monitoring Tool: CodeCarbon")
    print(f"{'='*70}\n")

    if args.num_prompts < len(prompts):
        sampled_prompts = prompts[:args.num_prompts]
    else:
        sampled_prompts = prompts

    print(f"Using {len(sampled_prompts)} prompts\n")

    print("Initializing vLLM...")

    with OfflineEmissionsTracker(country_iso_code="GBR") as tracker:
        llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            block_size=args.block_size,
            max_num_seqs=args.batch_size,
            disable_log_stats=False,
            max_num_batched_tokens=8192,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=args.enable_chunked_prefill,  
            enable_prefix_caching=args.enable_prefix_caching  
        )
        start_time = time.time()
        outputs = llm.generate(sampled_prompts, sampling_params)
        elapsed_time = time.time() - start_time
    
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_output_tokens / elapsed_time

    print(f"\nThroughput: {throughput:.2f} tokens/sec")
    print(f"Requests/sec: {len(outputs) / elapsed_time:.2f}")
    print(f"Total time: {elapsed_time:.2f}s")



if __name__ == "__main__":
    main()
