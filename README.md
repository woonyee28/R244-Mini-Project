# vLLM Multi-Objective Bayesian Optimization

A toolkit for optimizing vLLM inference configurations using multi-objective Bayesian optimization. This project helps find Pareto-optimal configurations that balance **throughput** (tokens/second) and **energy efficiency** (Joules/token).

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPUs
- NVIDIA drivers with CUDA support

### Dependencies

```bash
pip install vllm torch codecarbon ax-platform botorch datasets plotly
```
Alternatively, check out requirements.txt for reproducibity.

## Usage

### 1. Single Evaluation

Run a single vLLM evaluation with specific configuration:

```bash
python vllm_evaluation.py \
    --num_prompts 100 \
    --tensor_parallel_size 2 \
    --gpus "2,3" \
    --block_size 32 \
    --max_num_seqs 256 \
    --enable_chunked_prefill \
    --enable_prefix_caching
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--block_size` | 16 | Block size for Paged Attention |
| `--max_num_seqs` | 1024 | Maximum batch size |
| `--max_num_batched_tokens` | 8192 | Maximum batched tokens |
| `--num_prompts` | 1000 | Number of prompts to process |
| `--tensor_parallel_size` | 1 | GPUs for tensor parallelism |
| `--pipeline_parallel_size` | 1 | GPUs for pipeline parallelism |
| `--data_parallel_size` | 1 | GPUs for data parallelism |
| `--gpus` | "4" | Comma-separated GPU IDs |
| `--enable_chunked_prefill` | False | Enable chunked prefill |
| `--enable_prefix_caching` | False | Enable prefix caching |
| `--dtype` | "auto" | Data type (auto, float16, bfloat16) |
| `--enable_profiling` | False | Enable vLLM profiler |

### 2. Multi-Objective Optimization

Run Bayesian optimization to find Pareto-optimal configurations:

```bash
python optimize_vllm.py \
    --n_initial 3 \
    --n_iterations 10 \
    --num_prompts 1000 \
    --gpus "1,2,3,4" \
    --output_dir ./botorch_results
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--n_initial` | 10 | Initial random exploration trials |
| `--n_iterations` | 30 | Bayesian optimization iterations |
| `--num_prompts` | 100 | Prompts per evaluation |
| `--gpus` | "2,3,4,5" | Available GPU IDs |
| `--output_dir` | "./botorch_results" | Output directory |
| `--model` | "mistralai/Mistral-7B-Instruct-v0.1" | Model to optimize |

### 3. Simple CodeCarbon Evaluation

Quick evaluation with energy tracking:

```bash
python vllm_codecarbon.py \
    --tensor_parallel_size 1 \
    --pipeline_parallel_size 1 \
    --gpus "2" \
    --num_prompts 1000
```

### 4. NVIDIA Nsight Systems Profiling

Profile vLLM inference for detailed performance analysis:

```bash
nsys profile \
    -o vllm_profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    python vllm_evaluation.py \
        --block_size 32 \
        --max_num_seqs 256 \
        --tensor_parallel_size 4 \
        --gpus "4,5,6,7"
```

## Output

### Optimization Results

After optimization completes, results are saved to the output directory:

- **`trials_data.csv`**: Complete trial history with parameters and metrics
- **`pareto_optimal.json`**: Pareto-optimal configurations
- **`ax_experiment.json`**: Full Ax experiment state (can be resumed)

### Metrics

- **Throughput**: Tokens generated per second (tokens/s)
- **Energy per token**: Energy consumption per token (J/token)
- **Emissions**: CO2 emissions (kg CO2)
- **Total energy**: Total energy consumed (kWh)

## Dataset

The project uses the [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset for evaluation prompts.

## Model

Default model: `mistralai/Mistral-7B-Instruct-v0.1`

You can specify a different model using the `--model` argument in `optimize_vllm.py`.

## References

- [vLLM](https://github.com/vllm-project/vllm)
- [Ax](https://ax.dev/) 
- [BoTorch](https://botorch.org/)
- [CodeCarbon](https://codecarbon.io/) 
