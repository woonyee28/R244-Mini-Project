#!/bin/bash

commands=(
    "python vllm_evaluation.py --num_prompts 100 --block_size 48 --max_num_seqs 91 --max_num_batched_tokens 6596 --tensor_parallel_size 1 --enable_chunked_prefill --enable_prefix_caching"
    "python vllm_evaluation.py --num_prompts 100 --block_size 80 --max_num_seqs 154 --max_num_batched_tokens 9273 --pipeline_parallel_size 3 --enable_chunked_prefill"
    "python vllm_evaluation.py --num_prompts 100 --tensor_parallel_size 4 --enable_chunked_prefill"
    "python vllm_evaluation.py --num_prompts 100 --block_size 32 --max_num_seqs 256 --max_num_batched_tokens 12288 --tensor_parallel_size 4 --enable_chunked_prefill --enable_prefix_caching"
    "python vllm_evaluation.py --num_prompts 100 --block_size 32 --max_num_seqs 128 --max_num_batched_tokens 12288 --tensor_parallel_size 1 --enable_prefix_caching"
)

required_gpus=(1 3 4 4 1)

NUM_RUNS=5
MEMORY_THRESHOLD=10
MAX_RETRIES=5
RETRY_DELAY=30

get_free_gpus() {
    local needed=$1
    local free_gpus=()
    
    while IFS=',' read -r idx mem_used; do
        idx=$(echo "$idx" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        if [ "$mem_used" -lt "$MEMORY_THRESHOLD" ]; then
            free_gpus+=("$idx")
        fi
        if [ "${#free_gpus[@]}" -ge "$needed" ]; then
            break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    
    if [ "${#free_gpus[@]}" -ge "$needed" ]; then
        local IFS=','
        echo "${free_gpus[*]}"
    fi
}

for i in "${!commands[@]}"; do
    echo "========================================"
    echo "Command $((i+1)): ${commands[$i]}"
    echo "Required GPUs: ${required_gpus[$i]}"
    echo "========================================"
    
    run=1
    retries=0
    while [ "$run" -le "$NUM_RUNS" ]; do
        echo "--- Run $run of $NUM_RUNS ---"
        
        while true; do
            gpu_ids=$(get_free_gpus "${required_gpus[$i]}")
            [ -n "$gpu_ids" ] && break
            echo "Waiting for ${required_gpus[$i]} free GPU(s)..."
            sleep 30
        done
        
        echo "Using GPUs: $gpu_ids"
        
        output=$(eval "${commands[$i]} --gpus='${gpu_ids}'" 2>&1)
        echo "$output"
        
        if echo "$output" | grep -q "Error during evaluation\|RuntimeError\|ValueError\|CUDA out of memory\|Free memory on device"; then
            retries=$((retries + 1))
            if [ "$retries" -ge "$MAX_RETRIES" ]; then
                echo "MAX RETRIES ($MAX_RETRIES) reached for run $run - skipping to next run"
                run=$((run + 1))
                retries=0
            else
                echo "ERROR DETECTED - retrying run $run in $RETRY_DELAY seconds (retry $retries/$MAX_RETRIES)"
                sleep $RETRY_DELAY
            fi
        else
            echo "Run $run completed successfully"
            run=$((run + 1))
            retries=0
        fi
    done
done

echo "All runs completed!"