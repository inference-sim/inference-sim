import argparse
import json
from optimizer import InferenceSimOptimizer, parse_optimizer_config

def train_optimizer(config, n_trials: int, train_results_file: str):
    optimizer = InferenceSimOptimizer(**config)

    # Run optimization
    if n_trials is not None:
        optimizer.optimize(n_trials=n_trials)
    else:
        optimizer.optimize(n_trials=optimizer.n_trials)

    # Extract scaled parameters
    best_params = optimizer.get_best_params()
    scaling = optimizer.scaling
    best_params['sum_decode_tokens'] = best_params['sum_decode_tokens'] * scaling['sum_decode_tokens']
    best_params['sum_prefill_tokens'] = best_params['sum_prefill_tokens'] * scaling['sum_prefill_tokens']
    best_params['num_prefills'] = best_params['num_prefills'] * scaling['num_prefills']
    # intercept = best_params['intercept'] * self.scaling['intercept']
    best_params['step_constant'] = best_params['step_constant'] * scaling['step_constant']
    best_params['vllm_overhead'] = best_params['vllm_overhead'] * scaling['vllm_overhead']
    # Evaluate the optimizer
    optimizer.evaluate(best_params=best_params)

    # Save study results if requested
    optimizer.save_study(train_results_file)

def test_optimizer(config, eval_params_file):

    # load the last set of coefficients from params file
    with open(eval_params_file, 'r') as f:
        eval_params_all = json.load(f)
    coeff_keys = ["sum_decode_tokens", "sum_prefill_tokens", "num_prefills", "sum_decode_tokenss2", 
                  "sum_decode_tokensmsumprefill_tokens", "sum_decode_tokensmmaxprefill_tokens", 
                  "sum_decode_tokensmnumprefills", "intercept", "step_constant", "vllm_overhead"]
    params = {k: eval_params_all[-1][k] for k in coeff_keys}
    print (params)
    optimizer = InferenceSimOptimizer(**config)
    optimizer.evaluate(best_params=params)

def main():
    """
    Main function to run the inference simulation optimizer.
    """
    parser = argparse.ArgumentParser(description="Run Inference Simulation Optimizer")
    parser.add_argument(
        "--config_file",
        type=str,
        default="optimizer_config.yaml",
        help="Path to the optimizer configuration file."
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of iters to run optimizer for"
    )

    parser.add_argument(
        "--train_results_file",
        type=str,
        default="./results/optimization_results.json",
        help="Save the training results to this file."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train/test mode for optimizer"
    )

    parser.add_argument(
        "--test_coeffs_file",
        type=str,
        default="./results/optimization_results.json",
        help="File containing coefficients to evaluate the optimizer on"
    )
    
    args = parser.parse_args()

    # Parse config and initialize optimizer
    config = parse_optimizer_config(args.config_file)

    if args.mode == "train":
        train_optimizer(config, args.n_trials, args.train_results_file)
    else:
        test_optimizer(config, args.test_coeffs_file)

if __name__ == "__main__":
    main()
