import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")

def predict_with_adapter_directly(
    ax_client,
    params: Dict[str, Any]
) -> Optional[Dict[str, Dict[str, float]]]:
    from ax.core.observation import ObservationFeatures

    gs = ax_client.generation_strategy
    current_node = gs._curr
    adapter = getattr(current_node, '_fitted_adapter', None)
    
    if adapter is None:
        print("  Fitting model by generating a trial...")
        try:
            _ = ax_client.get_next_trial()
            current_node = gs._curr
            adapter = getattr(current_node, '_fitted_adapter', None)
        except Exception as e:
            print(f"  Could not fit model: {e}")
            return None
    
    obs_features = [ObservationFeatures(parameters=params)]
    f_pred, cov_pred = adapter.predict(obs_features)
    expected_metrics = set(ax_client.experiment.metrics.keys())

    if set(f_pred.keys()) == expected_metrics:
        results = {}
        for metric_name in expected_metrics:
            mean_val = f_pred[metric_name][0]
            var_val = cov_pred[metric_name][metric_name][0]
            results[metric_name] = {
                'mean': float(mean_val),
                'std': float(np.sqrt(var_val))
            }
        return results
    else:
        print(f"  WARNING: f_pred keys don't match expected metrics!")
        print(f"  f_pred keys: {set(f_pred.keys())}")
        print(f"  This is the Ax bug - trying BoTorch fallback...")
        return None
            


def safe_predict(json_path: str, params: Dict[str, Any]) -> Optional[Dict[str, Dict[str, float]]]:
    from ax.service.ax_client import AxClient
    
    print(f"\nLoading experiment from {json_path}")
    ax_client = AxClient.load_from_json_file(json_path)
    
    experiment = ax_client.experiment
    metrics = list(experiment.metrics.keys())
    exp_params = set(experiment.search_space.parameters.keys())
    
    print(f"  Metrics: {metrics}")
    print(f"  Parameters: {list(exp_params)}")
    
    filtered_params = {}
    for param_name in exp_params:
        if param_name in params:
            filtered_params[param_name] = params[param_name]
        else:
            param_obj = experiment.search_space.parameters[param_name]
            if hasattr(param_obj, 'value'):  
                filtered_params[param_name] = param_obj.value
            elif hasattr(param_obj, 'lower') and hasattr(param_obj, 'upper'):
                filtered_params[param_name] = (param_obj.lower + param_obj.upper) / 2
    
    print(f"  Using parameters: {filtered_params}")
    
    result = predict_with_adapter_directly(ax_client, filtered_params)
    if result:
        return result
    
    return None

test_params_list = [
    {
        "block_size": 64.0,
        "max_num_seqs": 64.0,
        "max_num_batched_tokens": 8192.0,
        "tensor_parallel_size": 4.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 1.0,
        "enable_prefix_caching": 0.0,
        "dtype_idx": 0.0,
        "dtype": "auto",
    },
    {
        "block_size": 64.0,
        "max_num_seqs": 64.0,
        "max_num_batched_tokens": 8192.0,
        "tensor_parallel_size": 2.0,
        "pipeline_parallel_size": 2.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 0.0,
        "enable_prefix_caching": 1.0,
        "dtype_idx": 1.0,
        "dtype": "float16",
    },
    {
        "block_size": 64.0,
        "max_num_seqs": 256.0,
        "max_num_batched_tokens": 4096.0,
        "tensor_parallel_size": 2.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 0.0,
        "enable_prefix_caching": 0.0,
        "dtype_idx": 2.0,
        "dtype": "bfloat16",
    },
    {
        "block_size": 128.0,
        "max_num_seqs": 128.0,
        "max_num_batched_tokens": 12288.0,
        "tensor_parallel_size": 4.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 0.0,
        "enable_prefix_caching": 0.0,
        "dtype_idx": 2.0,
        "dtype": "bfloat16",
    },
    {
        "block_size": 32.0,
        "max_num_seqs": 64.0,
        "max_num_batched_tokens": 4096.0,
        "tensor_parallel_size": 2.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 1.0,
        "enable_prefix_caching": 1.0,
        "dtype_idx": 1.0,
        "dtype": "float16",
    },
    {
        "block_size": 64.0,
        "max_num_seqs": 128.0,
        "max_num_batched_tokens": 12288.0,
        "tensor_parallel_size": 2.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 1.0,
        "enable_prefix_caching": 0.0,
        "dtype_idx": 2.0,
        "dtype": "bfloat16",
    },
    {
        "block_size": 128.0,
        "max_num_seqs": 256.0,
        "max_num_batched_tokens": 8192.0,
        "tensor_parallel_size": 2.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 0.0,
        "enable_prefix_caching": 1.0,
        "dtype_idx": 1.0,
        "dtype": "float16",
    },
    {
        "block_size": 32.0,
        "max_num_seqs": 256.0,
        "max_num_batched_tokens": 8192.0,
        "tensor_parallel_size": 4.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 0.0,
        "enable_prefix_caching": 0.0,
        "dtype_idx": 2.0,
        "dtype": "bfloat16",
    },
    {
        "block_size": 64.0,
        "max_num_seqs": 64.0,
        "max_num_batched_tokens": 12288.0,
        "tensor_parallel_size": 2.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 1.0,
        "enable_prefix_caching": 1.0,
        "dtype_idx": 1.0,
        "dtype": "float16",
    },
    {
        "block_size": 128.0,
        "max_num_seqs": 128.0,
        "max_num_batched_tokens": 4096.0,
        "tensor_parallel_size": 4.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 1.0,
        "enable_prefix_caching": 0.0,
        "dtype_idx": 1.0,
        "dtype": "float16",
    },
]

if __name__ == "__main__":
    # test configuration
    test_params = {
        "block_size": 64.0,
        "max_num_seqs": 128.0,
        "max_num_batched_tokens": 8192.0,
        "tensor_parallel_size": 4.0,
        "pipeline_parallel_size": 1.0,
        "data_parallel_size": 1,
        "enable_chunked_prefill": 0.0,
        "enable_prefix_caching": 0.0,
        "dtype_idx": 0.0,
        "dtype": "auto",  # for discrete experiment
    }
    
    print("=" * 70)
    print("Testing Ax Prediction Workaround")
    print("=" * 70)
    
    experiment_paths = [
        ("Continuous", "botorch_result_continuous/ax_experiment.json"),
        ("Discrete", "botorch_result_discrete/ax_experiment.json"),
    ]
    
    for test_params in test_params_list:
        print(f"\n{'='*70}")
        print(f"Test Params: {test_params}")
        print("=" * 70)
        
        for name, path in experiment_paths:
            print(f"\n  Testing: {name}")
            
            if not Path(path).exists():
                print(f"    Skipping (file not found: {path})")
                continue
            
            result = safe_predict(path, test_params.copy())
            
            if result:
                print(f"\n    Final Results:")
                for metric, values in result.items():
                    note = values.get('note', '')
                    print(f"      {metric}: {values['mean']:.4f} ± {values['std']:.4f} {note}")
            else:
                print(f"\n    Could not get predictions for {name}")