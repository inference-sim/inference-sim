import argparse
from optimizer import InferenceSimOptimizer, parse_optimizer_config

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
        help="Optional: Number of optimization trials to run. If not set, uses default from config."
    )
    parser.add_argument(
        "--save_study",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the study results to a file."
    )
    
    args = parser.parse_args()

    # Parse config and initialize optimizer
    config = parse_optimizer_config(args.config_file)
    optimizer = InferenceSimOptimizer(**config)

    # Run optimization
    if args.n_trials is not None:
        optimizer.optimize(n_trials=args.n_trials)
    else:
        optimizer.optimize(n_trials=optimizer.n_trials)

    # Evaluate the optimizer
    optimizer.evaluate()

    # Save study results if requested
    if args.save_study:
        optimizer.save_study()

if __name__ == "__main__":
    main()
