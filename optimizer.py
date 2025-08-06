import subprocess
import numpy as np
import optuna
import optunahub
from typing import Dict, Optional, Union
import os
import sys
import json
import yaml

def parse_optimizer_config(config_file: str):
    """
    Parses the optimizer configuration from a YAML file.

    Args:
        config_file: Path to the YAML configuration file.

    Returns:
        A dictionary of parameters to be passed to the InferenceSimOptimizer.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return {
        'pbounds': config.get('pbounds'),
        'scaling': config.get('scaling'),
        'tokens_dir': config.get('tokens_dir'),
        'sim_dir': config.get('sim_dir'),
        'vllm_dir': config.get('vllm_dir'),
        'kv_blocks': config.get('kv_blocks'),
        'error_function': config.get('error_function'),
        'train_config': config.get('train_config'),
        'test_config': config.get('test_config'),
        'seed': config.get('seed', 42)  # Default seed if not specified
    }
    

class InferenceSimOptimizer:
    """
    A class for black box optimization of inference simulation parameters.
    
    This class provides an easy interface to configure, run, and evaluate
    optimization experiments for inference simulation models.
    """
    
    def __init__(
        self,
        pbounds: Optional[Dict] = None,
        scaling: Optional[Dict] = None,
        tokens_dir: str = "data/output_tokens_2025-06-30_tokenized.json",
        sim_dir: str = "./results/qwen2-0.5b/sweep_params",
        vllm_dir: str = "../pytools/experiments/profiling/qwen2-0.5b/sweep",
        kv_blocks: int = 94060,
        error_function: str = "mape",
        train_config: Optional[Dict] = None,
        test_config: Optional[Dict] = None,
        seed: int = 42
    ):
        """
        Initialize the InferenceSimOptimizer.
        
        Args:
            pbounds: Parameter bounds for optimization variables
            scaling: Scaling factors for parameters
            tokens_dir: Path to tokenized data file
            sim_dir: Directory for simulation outputs
            vllm_dir: Directory containing vLLM reference files
            kv_blocks: Number of KV cache blocks
            error_function: Error metric to use ('mse' or 'mape')
            train_config: Configuration for training/optimization
            test_config: Configuration for testing/evaluation
        """
        # Set default parameter bounds
        self.pbounds = pbounds or {
            'sum_decode_tokens': (0, 1),
            'sum_prefill_tokens': (0, 1),
            'num_prefills': (0, 1),
            'sum_decode_tokenss2': (0, 1),
            'sum_decode_tokensmsumprefill_tokens': (0, 1),
            'sum_decode_tokensmmaxprefill_tokens': (0, 1),
            'sum_decode_tokensmnumprefills': (0, 1),
            'intercept': (0, 1),
            'step_constant': (0, 1),
            'vllm_overhead': (0, 1)
        }
        
        # Set default scaling factors
        self.scaling = scaling or {
            'sum_decode_tokens': 0.0001,
            'sum_prefill_tokens': 0.0001,
            'num_prefills': 0.0001,
            'sum_decode_tokenss2': 0.0001,
            'sum_decode_tokensmsumprefill_tokens': 0.0001,
            'sum_decode_tokensmmaxprefill_tokens': 0.0001,
            'sum_decode_tokensmnumprefills': 0.0001,
            'intercept': 0.01,
            'step_constant': 2000,
            'vllm_overhead': 6000
        }
        
        # Set default training configuration
        self.train_config = train_config or {
            'num_prompts': [100, 400, 1600],
            'request_rate': [8, 64],
            'temperature': [0.0],
            'max_num_batched_tokens': [256, 1024],
            'long_prefill_token_threshold': [16, 1024],
            'datasets': [{'name': 'sharegpt', 'path': 'ShareGPT_V3_unfiltered_cleaned_split.json'}]
        }
        
        # Set default test configuration
        self.test_config = test_config or {
            'num_prompts': [100, 200, 400, 800, 1600],
            'request_rate': [4, 8, 16, 32, 64, 128],
            'temperature': [0.0],
            'max_num_batched_tokens': [256, 512, 1024, 2048, 4096, 8192],
            'long_prefill_token_threshold': [16, 32, 64, 128, 256, 512, 1024],
            'datasets': [{'name': 'sharegpt', 'path': 'ShareGPT_V3_unfiltered_cleaned_split.json'}]
        }
        
        # Store configuration
        self.tokens_dir = tokens_dir
        self.sim_dir = sim_dir
        self.vllm_dir = vllm_dir
        self.kv_blocks = kv_blocks
        self.error_function = error_function
        
        # Initialize study placeholder
        self.study = None
        self.train_score = None
        self.eval_score = None
        self.seed = seed
        # Print initialization summary
        self._print_init_summary()
        
        # Import utility functions (these would need to be imported from your existing code)
        # For now, we'll define them as methods within the class
    
    def _run_command(self, command):
        """Execute a command and return stderr output."""
        try:
            result = subprocess.run(command, check=True, capture_output=True)
            return result.stderr.decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(f"Error executing command '{' '.join(command)}': {e}")
            print(f"Output: {e.output}")
            print(f"Error output: {e.stderr}")
            raise
    
    def _parse_outputs_from_file(self, sim_dir: str, vllm_dir: str, sweep_configs: dict):
        """Parse simulation and vLLM outputs from files."""
        # This would need to import from your existing pytools.experiments.utils
        # For now, return placeholder values
        # Ensure pytools is in the path
        parent_dir = os.path.dirname(os.getcwd())
        sys.path.insert(0, parent_dir)

        from pytools.experiments.utils import find_matching_experiments, parse_sim_results, parse_vllm_results
        
        matching_pairs = find_matching_experiments(sim_dir, vllm_dir, sweep_configs)
        
        if len(matching_pairs) == 0:
            print(f"No matching pairs found. Please check the directories and configurations: {sim_dir} and {vllm_dir}, {sweep_configs}")
            return ([], [])
        
        vllm_ttfts = []
        vllm_tpots = []
        vllm_med_ttfts = []
        vllm_med_tpots = []
        vllm_mean_ttfts = []
        vllm_mean_tpots = []
        vllm_p99_ttfts = []
        vllm_p99_tpots = []
        
        sim_ttfts = []
        sim_tpots = []
        sim_med_ttfts = []
        sim_med_tpots = []
        sim_mean_ttfts = []
        sim_mean_tpots = []
        sim_p99_ttfts = []
        sim_p99_tpots = []
        
        for sim_file, vllm_file, config in matching_pairs:
            sim_results = parse_sim_results(sim_file)
            sim_ttfts.extend(sim_results['ttfts'])
            sim_tpots.extend(sim_results['tpots'])
            sim_med_ttfts.append(sim_results['ttft_median'])
            sim_med_tpots.append(sim_results['tpot_median'])
            sim_mean_ttfts.append(sim_results['ttft_mean'])
            sim_mean_tpots.append(sim_results['tpot_mean'])
            sim_p99_ttfts.append(sim_results['ttft_p99'])
            sim_p99_tpots.append(sim_results['tpot_p99'])
            
            vllm_results = parse_vllm_results(vllm_file)
            vllm_ttfts.extend(vllm_results['ttfts'])
            vllm_tpots.extend(vllm_results['tpots'])
            vllm_med_ttfts.append(vllm_results['ttft_median'])
            vllm_med_tpots.append(vllm_results['tpot_median'])
            vllm_mean_ttfts.append(vllm_results['ttft_mean'])
            vllm_mean_tpots.append(vllm_results['tpot_mean'])
            vllm_p99_ttfts.append(vllm_results['ttft_p99'])
            vllm_p99_tpots.append(vllm_results['tpot_p99'])
        
        vllm_values = (np.array(vllm_med_ttfts), np.array(vllm_med_tpots),
                       np.array(vllm_mean_ttfts), np.array(vllm_mean_tpots),
                       np.array(vllm_p99_ttfts), np.array(vllm_p99_tpots),
                       np.array(vllm_ttfts), np.array(vllm_tpots))
        
        sim_values = (np.array(sim_med_ttfts), np.array(sim_med_tpots),
                      np.array(sim_mean_ttfts), np.array(sim_mean_tpots),
                      np.array(sim_p99_ttfts), np.array(sim_p99_tpots),
                      np.array(sim_ttfts), np.array(sim_tpots))
        
        return sim_values, vllm_values
    
    def _get_error_mse(self, sim_values, vllm_values):
        """Calculate Mean Squared Error."""
        if len(sim_values) != len(vllm_values) or len(sim_values) == 0:
            return np.inf
        
        mse = 0
        for i in range(len(sim_values)):
            if len(sim_values[i]) != len(vllm_values[i]) or len(sim_values[i]) == 0:
                return np.inf
            mse += np.mean((sim_values[i] - vllm_values[i]) ** 2)
        
        return mse / len(sim_values)
    
    def _get_error_mape(self, sim_values, vllm_values):
        """Calculate Mean Absolute Percentage Error."""
        mape = 0
        num_features = 0
        
        for i in range(len(sim_values)):
            sim = sim_values[i]
            vllm = vllm_values[i]
            
            if len(sim) < len(vllm):
                print(f"Warning: Length of simulation values ({len(sim)}) is less than vLLM values ({len(vllm)}) for feature {i}.")
                return np.inf
            
            if len(vllm) < len(sim):
                continue
            
            if len(sim) == 0:
                continue
            
            non_zero_mask = vllm > 0
            if np.sum(non_zero_mask) == 0:
                continue
            
            mape += np.mean(np.abs((sim[non_zero_mask] - vllm[non_zero_mask]) / vllm[non_zero_mask]))
            num_features += 1
        
        if num_features == 0:
            return np.inf
        
        return mape / num_features
    
    def _print_init_summary(self):
        """Print initialization summary."""
        print("=" * 60)
        print("INFERENCE SIM OPTIMIZER INITIALIZED")
        print("=" * 60)
        print(f"Parameter Bounds: {self.pbounds}")
        print(f"Scaling Factors: {self.scaling}")
        print(f"KV Blocks: {self.kv_blocks}")
        print(f"Tokens Directory: {self.tokens_dir}")
        print(f"Simulation Directory: {self.sim_dir}")
        print(f"vLLM Directory: {self.vllm_dir}")
        print(f"Error Function: {self.error_function}")
        print(f"Train Config: {self.train_config}")
        print(f"Test Config: {self.test_config}")
        print(f"Seed: {self.seed}")
        print("=" * 60)
    
    def _get_error(self, sim_values, vllm_values):
        """Get error using the specified error function."""
        if self.error_function == "mse":
            return self._get_error_mse(sim_values, vllm_values)
        elif self.error_function == "mape":
            return self._get_error_mape(sim_values, vllm_values)
        else:
            raise ValueError(f"Unknown error function: {self.error_function}")
    
    def _create_objective_function(self, sweep_configs, sim_dir):
        """Create the objective function for optimization."""
        def objective(trial):
            # Extract parameters from trial
            sum_decode_tokens = trial.suggest_float('sum_decode_tokens', *self.pbounds['sum_decode_tokens']) * self.scaling['sum_decode_tokens']
            sum_prefill_tokens = trial.suggest_float('sum_prefill_tokens', *self.pbounds['sum_prefill_tokens']) * self.scaling['sum_prefill_tokens']
            num_prefills = trial.suggest_float('num_prefills', *self.pbounds['num_prefills']) * self.scaling['num_prefills']
            # intercept = trial.suggest_float('intercept', *self.pbounds['intercept']) * self.scaling['intercept']
            step_constant = trial.suggest_float('step_constant', *self.pbounds['step_constant']) * self.scaling['step_constant']
            vllm_overhead = trial.suggest_float('vllm_overhead', *self.pbounds['vllm_overhead']) * self.scaling['vllm_overhead']
            
            # Set unused parameters to 0 for simplicity
            sum_decode_tokenss2 = 0
            sum_decode_tokensmsumprefill_tokens = 0
            sum_decode_tokensmmaxprefill_tokens = 0
            sum_decode_tokensmnumprefills = 0
            
            # Create coefficients array
            coefficients = [sum_decode_tokens, sum_prefill_tokens, 0, num_prefills, 
                          sum_decode_tokenss2, sum_decode_tokensmsumprefill_tokens, 
                          sum_decode_tokensmmaxprefill_tokens, sum_decode_tokensmnumprefills, 
                          0, 0, 0, 0, 0, 0, 0]
            coefficients_str = ','.join(map(str, coefficients))
            
            # Build command
            command = ["python", "request_rate_sweep.py", "--rates"]
            command.extend(list(map(str, sweep_configs["request_rate"])))
            command.append("--long_prefill_token_thresholds")
            command.extend(list(map(str, sweep_configs["long_prefill_token_threshold"])))
            command.append("--max_num_batched_tokens")
            command.extend(list(map(str, sweep_configs["max_num_batched_tokens"])))
            command.append("--num_requests")
            command.extend(list(map(str, sweep_configs["num_prompts"])))
            command.extend([
                "--input_filename", f"{self.tokens_dir}",
                "--regression_coeffs", f'{coefficients_str}',
                "--schedule_time", f"{str(int(step_constant))}",
                "--update_time", f"{str(int(0))}",
                "--queue_overhead_time", f"{str(int(0))}",
                "--vllm_overhead_time", f"{str(int(vllm_overhead))}",
                "--total_kv_blocks", f"{str(self.kv_blocks)}",
                "--output_dir", sim_dir
            ])
            
            # Run simulation
            self._run_command(command)
            
            # Parse results and calculate error
            sim_values, vllm_values = self._parse_outputs_from_file(sim_dir, self.vllm_dir, sweep_configs)
            error = self._get_error(sim_values, vllm_values)
            
            return error
        
        return objective
    
    def optimize(self, n_trials: int = 100, sampler: str = "implicit_natural_gradient"):
        """
        Run optimization study.
        
        Args:
            n_trials: Number of optimization trials to run, defaults to 100
            sampler: Sampler to use for optimization, defaults to "implicit_natural_gradient"
            seed: Random seed for reproducibility, defaults to 42
        """
        # Print optimization start summary
        print("=" * 60)
        print("STARTING OPTIMIZATION")
        print("=" * 60)
        print(f"Number of Trials: {n_trials}")
        print(f"Sampler: {sampler}")
        print(f"Simulation Directory: {self.sim_dir}")
        print(f"vLLM Directory: {self.vllm_dir}")
        print(f"Train Config: {self.train_config}")
        print(f"Error Function: {self.error_function}")
        print("=" * 60)
        
        # Reset eval score when starting new optimization
        self.eval_score = None
        
        # Create sampler
        if sampler == "implicit_natural_gradient":
            mod = optunahub.load_module("samplers/implicit_natural_gradient")
            sampler_obj = mod.ImplicitNaturalGradientSampler(seed=self.seed)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # Create study
        self.study = optuna.create_study(sampler=sampler_obj)
        
        # Create objective function
        objective_func = self._create_objective_function(self.train_config, self.sim_dir)
        
        # Run optimization
        self.study.optimize(objective_func, n_trials=n_trials)
        
        # Store train score
        self.train_score = self.study.best_value
        
        # Print optimization completion summary
        print("=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Best Training Error: {self.study.best_value}")
        print(f"Best Parameters: {self.study.best_params}")
        print("=" * 60)
    
    def get_best_params(self):
        """
        Get the best parameters from the current study.
        
        Returns:
            Dictionary of best parameters
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        return self.study.best_params
    
    def visualize_study(self):
        """Visualize the optimization study results."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        import optuna.visualization
        return optuna.visualization.plot_optimization_history(self.study)
    
    def evaluate(self, config: Optional[Dict] = None, sim_dir: Optional[str] = None) -> float:
        """
        Evaluate the optimized model on test configuration.
        
        Args:
            config: Test configuration to use (defaults to self.test_config)
            sim_dir: Simulation directory to use (defaults to self.sim_dir + "/test")
            
        Returns:
            Error value on test set
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        # Use provided config or default test config
        test_config = config or self.test_config
        test_sim_dir = sim_dir or (self.sim_dir)
        
        # Print evaluation information
        print("=" * 60)
        print("EVALUATION DETAILS")
        print("=" * 60)
        print(f"Train Config: {self.train_config}")
        print(f"Test Config: {test_config}")
        print(f"Train Score: {self.train_score}")
        print(f"vLLM Directory: {self.vllm_dir}")
        print(f"Simulation Directory (eval): {test_sim_dir}")
        print("=" * 60)
        
        # Get best parameters from study
        best_params = self.study.best_params
        
        # Extract scaled parameters
        sum_decode_tokens = best_params['sum_decode_tokens'] * self.scaling['sum_decode_tokens']
        sum_prefill_tokens = best_params['sum_prefill_tokens'] * self.scaling['sum_prefill_tokens']
        num_prefills = best_params['num_prefills'] * self.scaling['num_prefills']
        # intercept = best_params['intercept'] * self.scaling['intercept']
        step_constant = best_params['step_constant'] * self.scaling['step_constant']
        vllm_overhead = best_params['vllm_overhead'] * self.scaling['vllm_overhead']
        
        # Create coefficients array
        coefficients = [sum_decode_tokens, sum_prefill_tokens, 0, num_prefills, 
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        coefficients_str = ','.join(map(str, coefficients))
        
        # Build command
        command = ["python", "request_rate_sweep.py", "--rates"]
        command.extend(list(map(str, test_config["request_rate"])))
        command.append("--long_prefill_token_thresholds")
        command.extend(list(map(str, test_config["long_prefill_token_threshold"])))
        command.append("--max_num_batched_tokens")
        command.extend(list(map(str, test_config["max_num_batched_tokens"])))
        command.append("--num_requests")
        command.extend(list(map(str, test_config["num_prompts"])))
        command.extend([
            "--input_filename", f"{self.tokens_dir}",
            "--regression_coeffs", f'{coefficients_str}',
            "--schedule_time", f"{str(int(step_constant))}",
            "--update_time", f"{str(int(0))}",
            "--queue_overhead_time", f"{str(int(0))}",
            "--vllm_overhead_time", f"{str(int(vllm_overhead))}",
            "--total_kv_blocks", f"{str(self.kv_blocks)}",
            "--output_dir", test_sim_dir
        ])
        
        # Run simulation
        self._run_command(command)
        
        # Parse results and calculate error
        sim_values, vllm_values = self._parse_outputs_from_file(test_sim_dir, self.vllm_dir, test_config)
        error = self._get_error(sim_values, vllm_values)
        
        # Store eval score
        self.eval_score = error
        
        print(f"Evaluation Score: {error}")
        
        return error
    
    def save_study(self, filename: Optional[str] = None):
        """
        Save the current study results to a JSON file.

        Args:
            filename: Path to save the JSON file (defaults to "./results/optimization_results.json")
        """

        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        # If no evaluation has been done, run it first
        if self.eval_score is None:
            print("No evaluation score available. Running evaluation first...")
            self.evaluate()

        # Default filename
        if filename is None:
            filename = "./results/optimization_results.json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Get best parameters with scaling applied
        best_params = self.study.best_params
        result = {
            "sum_decode_tokens": best_params['sum_decode_tokens'] * self.scaling['sum_decode_tokens'],
            "sum_prefill_tokens": best_params['sum_prefill_tokens'] * self.scaling['sum_prefill_tokens'],
            "num_prefills": best_params['num_prefills'] * self.scaling['num_prefills'],
            "sum_decode_tokenss2": 0,
            "sum_decode_tokensmsumprefill_tokens": 0,
            "sum_decode_tokensmmaxprefill_tokens": 0,
            "sum_decode_tokensmnumprefills": 0,
            "intercept": 0,
            "step_constant": best_params['step_constant'] * self.scaling['step_constant'],
            "vllm_overhead": best_params['vllm_overhead'] * self.scaling['vllm_overhead'],
            "train_error": self.train_score,
            "eval_error": self.eval_score,
            "test_config": self.test_config,
            "train_config": self.train_config,
            "scaling": self.scaling,
            "vllm_path": self.vllm_dir
        }

        # Save as JSON (append if file exists, else create new list)
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        else:
            data = []

        data.append(result)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Study results saved to {filename}")
