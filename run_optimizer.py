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
        default="optimzer_config.yaml",
        help="Path to the optimizer configuration file."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of optimization trials to run."
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
    optimizer.optimize(n_trials=args.n_trials)

    # Evaluate the optimizer
    optimizer.evaluate()

    # Save study results if requested
    if args.save_study:
        optimizer.save_study()

if __name__ == "__main__":
    main()
