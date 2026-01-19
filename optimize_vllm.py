import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
import json
import argparse
from typing import List, Tuple, Dict

# reference: https://ax.dev/docs/0.5.0/tutorials/gpei_hartmann_service/
# more recently: https://ax.dev/docs/tutorials/getting_started/
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import init_notebook_plotting, render
import plotly.io as pio
from typing import List, Dict, Any, Optional

from vllm_evaluation import VLLMEvaluator, VLLMConfig, EvaluationResult


class VLLMOptimizer:
    """Multi-objective Bayesian Optimization for vLLM configuration"""
    
    def __init__(
        self,
        evaluator: VLLMEvaluator,
        parameters: List[Dict[str, Any]],
        gpu_ids: List[int],
        num_prompts: int = 500,
        output_dir: str = "./multiobj_result"
    ):
        """
        Args:
            evaluator: VLLMEvaluator instance
            bounds: torch.Tensor of shape (2, n_params) with lower and upper bounds
            param_names: List of parameter names
            gpu_ids: List of available GPU IDs
            num_prompts: Number of prompts to use for each evaluation
            output_dir: Directory to save results
        """
        self.evaluator = evaluator
        self.parameters = parameters
        self.gpu_ids = gpu_ids
        self.num_prompts = num_prompts
        self.output_dir = Path(output_dir)
        self.ax_client: Optional[AxClient] = None
        self.max_available_gpus = len(gpu_ids)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _discretize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert continuous GP values to valid vLLM discrete values"""
        """for cases where we use RANGE ax parameters instead of CHOICE parameters"""
        dtype_map = ['auto', 'float16', 'bfloat16']
        dtype_idx = int(round(params["dtype_idx"]))
        dtype_idx = max(0, min(2, dtype_idx))
        
        return {
            "block_size": int(round(params["block_size"] / 16) * 16),
            "max_num_seqs": int(round(params["max_num_seqs"])),
            "max_num_batched_tokens": int(round(params["max_num_batched_tokens"])),
            "tensor_parallel_size": int(round(params["tensor_parallel_size"])),
            "pipeline_parallel_size": int(round(params["pipeline_parallel_size"])),
            "data_parallel_size": params.get("data_parallel_size", 1),
            "enable_chunked_prefill": params["enable_chunked_prefill"] >= 0.5,
            "enable_prefix_caching": params["enable_prefix_caching"] >= 0.5,
            "dtype": dtype_map[dtype_idx],    
        }

    def _run_evaluation(self, parameter_values: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Ax objective function wrapper: Executes VLLMEvaluator and formats results.
        We assume multi-objective optimization: Maximize THROUGHPUT, Minimize ENERGY.
        """
        discrete_params = self._discretize_params(parameter_values)

        config = VLLMConfig(
            block_size=discrete_params["block_size"],
            max_num_seqs=discrete_params["max_num_seqs"],
            max_num_batched_tokens=discrete_params["max_num_batched_tokens"],
            tensor_parallel_size=discrete_params["tensor_parallel_size"],
            pipeline_parallel_size=discrete_params["pipeline_parallel_size"],
            data_parallel_size=discrete_params["data_parallel_size"],
            enable_chunked_prefill=discrete_params["enable_chunked_prefill"],
            enable_prefix_caching=discrete_params["enable_prefix_caching"],
            dtype=discrete_params["dtype"],
            )

        if config.total_gpus > self.max_available_gpus:
            print(f"Skipping trial: {config.total_gpus} GPUs required, only {self.max_available_gpus} available.")
            return {
                "throughput(token/s)": (0.0, 0.0),
                "energy(J/token)": (1e9, 0.0)
            }

        gpus_to_use = self.gpu_ids[:config.total_gpus]

        try:
            result = self.evaluator.evaluate(
                config=config,
                gpu_ids=gpus_to_use,
                num_prompts=self.num_prompts,
                verbose=False 
            )
    
            if result:
                return {
                    "throughput(token/s)": (result.throughput, 0.0),
                    "energy(J/token)": (result.energy_per_token, 0.0)
                }
            else:
                print("Evaluation failed, returning worst case metrics.")
                return {
                    "throughput(token/s)": (0.0, 0.0),
                    "energy(J/token)": (1e9, 0.0)
                }
        finally:
            import gc
            import torch
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            
            import time
            time.sleep(2)

    
    def run_optimization(self, n_initial: int, n_iterations: int):
        """Initializes and runs the Bayesian Optimization loop."""
        self.ax_client = AxClient()

        self.ax_client.create_experiment(
            name="vllm_multi_objective_tuning",
            parameters=self.parameters,
            objectives={
                "throughput(token/s)": ObjectiveProperties(minimize=False),
                "energy(J/token)": ObjectiveProperties(minimize=True)
            }
        )

        print("=" * 70)
        print(f"Starting Bayesian Optimization: {n_initial} random + {n_iterations} BO trials.")
        
        for i in range(n_initial):
            parameters, trial_index = self.ax_client.get_next_trial()
            raw_data = self._run_evaluation(parameters)
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            print(f"   Initial Trial {i+1}/{n_initial} completed. Throughput: {raw_data['throughput(token/s)'][0]:.2f}, Energy: {raw_data['energy(J/token)']}")

        for i in range(n_iterations):
            parameters, trial_index = self.ax_client.get_next_trial()
            raw_data = self._run_evaluation(parameters)
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            try:
                pareto_optimal = self.ax_client.get_pareto_optimal_parameters()
                pareto_size = len(pareto_optimal)  
            except Exception as e:
                print(f"Warning: Could not get Pareto frontier: {e}")
                pareto_size = "N/A"
            
            print(
                f"   BO Iteration {i+1}/{n_iterations} completed. "
                f"Throughput: {raw_data['throughput(token/s)'][0]:.2f}, "
                f"Energy: {raw_data['energy(J/token)'][0]:.6f}. "
                f"Pareto size: {pareto_size}."
            )

        print("=" * 70)
        print("Optimization Complete.")
        trials_df = self.ax_client.get_trials_data_frame()
        trials_df.to_csv(f"{self.output_dir}/trials_data.csv", index=False)
        best_arms = self.ax_client.get_pareto_optimal_parameters()
        print("=" * 70)
        print(f"\n📊 Found {len(best_arms)} Pareto-optimal configurations:")
        for arm_name, params in best_arms.items():
            print(f"  - {arm_name}: {params}")
        with open(self.output_dir / "pareto_optimal.json", "w") as f:
            json.dump(best_arms, f, indent=2)
        
        self.ax_client.save_to_json_file(filepath=str(self.output_dir / "ax_experiment.json"))
        
       
    

def main():
    ### Search Space:
    #    block_size: [32,64,128]
    #    max_num_seqs: [64,128,256]
    #    max_num_batched_tokens: int = [4096,8192,12288]
    #    tensor_parallel_size: int = [1,2]
    #    pipeline_parallel_size: int = [1,2]
    #    data_parallel_size: int = [1,2]
    #    enable_chunked_prefill: bool = [True, False]
    #    enable_prefix_caching: bool = [True, False]
    #    dtype: Optional[str] = [None, 'float16', 'bfloat16']
    parser = argparse.ArgumentParser(
        description='Multi-objective Bayesian Optimization for vLLM configuration'
    )
    parser.add_argument('--n_initial', type=int, default=10,
                       help='Number of initial random samples')
    parser.add_argument('--n_iterations', type=int, default=30,
                       help='Number of Bayesian Optimization iterations')
    parser.add_argument('--num_prompts', type=int, default=100,
                       help='Number of prompts per evaluation')
    parser.add_argument('--gpus', type=str, default='2,3,4,5',
                       help='Comma-separated GPU IDs')
    parser.add_argument('--output_dir', type=str, default='./botorch_results',
                       help='Output directory for results')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.1',
                       help='Model name')
    args = parser.parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    parameters = [
        {
            "name": "block_size",
            "type": "range",
            "bounds": [32.0, 128.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "max_num_seqs",
            "type": "range",
            "bounds": [64.0, 256.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "max_num_batched_tokens",
            "type": "range",
            "bounds": [4096.0, 12288.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "tensor_parallel_size",
            "type": "range",
            "bounds": [1.0, 4.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "pipeline_parallel_size",
            "type": "range",
            "bounds": [1.0, 4.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "data_parallel_size",
            "type": "fixed",
            "value": 1,
        },
        {
            "name": "enable_chunked_prefill",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "enable_prefix_caching",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "dtype_idx",
            "type": "range",
            "bounds": [0.0, 2.0],
            "value_type": "float",
        },
    ]
        
    evaluator = VLLMEvaluator(model_name=args.model)
    evaluator.load_dataset()
    
    optimizer = VLLMOptimizer(
        evaluator=evaluator,
        parameters=parameters,
        gpu_ids=gpu_ids,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir
    )
    
    optimizer.run_optimization(
        n_initial=args.n_initial,
        n_iterations=args.n_iterations
    )


if __name__ == "__main__":
    main()
