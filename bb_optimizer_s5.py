import os
import time
import threading
import concurrent.futures
from typing import Dict, Optional
import optuna
import optunahub
from optuna.samplers import TPESampler

from experiment_configs_scenario5 import config
from utils_s5 import get_unsaturated_exps, parse_vllm_metrics_to_json, run_go_binary

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)

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
        seed: int = 42,
        n_trials: int = 100
    ):
        """
        Initialize the InferenceSimOptimizer.
        
        Args:
            pbounds: Parameter bounds for optimization variables
            scaling: Scaling factors for parameters
        """
        # Set default parameter bounds
        self.pbounds = pbounds or {
            'beta0': (0, 1),
            'beta1': (0, 1),
            'beta2': (0, 1),
        }
        self.scaling = scaling or {
            'beta0': 1e-2,
            'beta1': 1e-4,
            'beta2': 1e-4
        }
        self.seed = seed
        if n_trials <= 0:
            raise ValueError("Number of trials must be a positive integer.")
        self.n_trials = n_trials
        self.metrics_lock = threading.Lock()
        self.model_name = config["MODEL"].split("/")[-1].replace(".", "_")
        vllm_results_folder = f"../vllm-data-collection/scenario4/results_server_side/{self.model_name}/train"
        self.unsaturated_exps = get_unsaturated_exps(vllm_results_folder)
        self.alpha_coeffs = config["QUEUING_COEFFS"][self.model_name] + config["FINISHED_COEFFS"][self.model_name]
        self.alpha_coeffs = list(map(str, self.alpha_coeffs))
        self.total_kv_blocks = config["TOTAL_KV_BLOCKS"][self.model_name]

        # get vllm ground truth metrics and save into a dict, 
        # to avoid processing every iteration
        self.all_vllm_metrics = {}
        for exp in self.unsaturated_exps:
            request_rate, spec, mbnt = exp["rr"], exp["spec"], exp["mbnt"]
            vllm_filename = f"vllm_{request_rate}r_{spec}_{mbnt}.json"
            exp_vllm_metrics = parse_vllm_metrics_to_json(vllm_results_folder, vllm_filename)
            self.all_vllm_metrics[f"{request_rate}r_{spec}_{mbnt}"] = exp_vllm_metrics
        
    def cost_function(self, vllm_metrics, sim_metrics):
        metric_names = ["Mean E2E(ms)", "Median E2E(ms)", "P99 E2E(ms)"]
        total_mape = 0
        for _, metric in enumerate(metric_names):
            mape = abs(sim_metrics[metric] - vllm_metrics[metric])/vllm_metrics[metric] * 100
            total_mape += mape
        return total_mape
        
    def per_thread_cost(self, spec, mbnt, request_rate, beta_coeffs):
        """
        Run simulator per experiment thread and obtain simulator results. 
        Compare against vllm ground truth metrics and return cost per experiment
        """
        # get vllm ground truth
        if f"{request_rate}r_{spec}_{mbnt}" in self.all_vllm_metrics:
            vllm_metrics = self.all_vllm_metrics[f"{request_rate}r_{spec}_{mbnt}"]
        else:
            return None
        
        # get sim metrics
        requests_folder = f"data/train/scenario4/{self.model_name}/{spec}/mbnt_{mbnt}/rr_{request_rate}"
        data_file_path = os.path.join(requests_folder, f"detailed_results_train_tokenized.json")
        args = {
            "rate": str(float(request_rate) / 1e6), "max-num-running-reqs": 8192, 
            "total-kv-blocks": self.total_kv_blocks, "max-num-scheduled-tokens": mbnt, 
            "block-size-in-tokens": 16, "horizon": "922337203685477580",
            "regression-coeffs": ','.join(beta_coeffs),
            "requests-file-path": data_file_path,
            "long-prefill-token-threshold": 0, "queuing-coeffs": ','.join(self.alpha_coeffs[:2]),
            "finished-coeffs": ','.join(self.alpha_coeffs[2:]), "log": "error"
        }
        args_list = ["run"]
        for key in args:
            args_list.extend([f"--{key}", str(args[key])])
        sim_metrics = run_go_binary(args_list, GO_BINARY_PATH, self.metrics_lock, self.model_name, spec, mbnt, request_rate)
        if not sim_metrics:
            return None
        cost = self.cost_function(vllm_metrics, sim_metrics)
        return cost

    def run_task_from_tuple(self, args_tuple):
        return self.per_thread_cost(*args_tuple)

    def multithreaded_obj(self, trial: optuna.trial.Trial):
        total_cost = 0.0
        tasks = []
        beta0 = trial.suggest_float('beta0', *self.pbounds['beta0']) * self.scaling['beta0']
        beta1 = trial.suggest_float('beta1', *self.pbounds['beta1']) * self.scaling['beta1']
        beta2 = trial.suggest_float('beta2', *self.pbounds['beta2']) * self.scaling['beta2']
        beta_coeffs = [beta0, beta1, beta2]
        beta_coeffs = list(map(str, beta_coeffs))

        for exp in self.unsaturated_exps:
            tasks.append((exp["spec"], exp["mbnt"], exp["rr"], beta_coeffs))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_costs = executor.map(self.run_task_from_tuple, tasks)
            valid_costs = [cost for cost in all_costs if cost is not None]
            total_cost = sum(valid_costs)
        return total_cost


    def optimize(self, sampler: str = ""):
        """
        Run optimization study.
        
        Args:
            n_trials: Number of optimization trials to run, defaults to 100
            sampler: Sampler to use for optimization, defaults to "TPESampler"
        """
        print("=" * 60)
        print("STARTING OPTIMIZATION")
        print("=" * 60)
        print(f"Number of Trials: {self.n_trials}")
        print(f"Sampler: {sampler}")
        print("=" * 60)
        
        if sampler == "implicit_natural_gradient":
            mod = optunahub.load_module("samplers/implicit_natural_gradient")
            sampler_obj = mod.ImplicitNaturalGradientSampler(seed=self.seed)
        else:
            sampler_obj = TPESampler(seed=self.seed)
        
        self.study = optuna.create_study(sampler=sampler_obj, 
                                         direction="minimize", 
                                         load_if_exists=False,
                                         storage=None)  
        self.study.optimize(self.multithreaded_obj, n_trials=self.n_trials)    
        self.train_score = self.study.best_value
        
        print("=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Best Training Error: {self.study.best_value}")
        print(f"Best Parameters:")
        for param in self.study.best_params:
            print(f"{param}: {self.study.best_params[param] * self.scaling[param]}")
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
    
optimizer = InferenceSimOptimizer()
optimizer.optimize(sampler="implicit_natural_gradient")
optimizer.visualize_study()