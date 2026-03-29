#!/usr/bin/env python3
"""
Monitor inner loop Bayesian optimization progress in real-time.

Usage:
    python scripts/monitor_optimization.py --iteration N [--interval 10] [--n-trials 50]

Press Ctrl+C to exit.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional


def load_results(filepath: str) -> Optional[dict]:
    """Load results JSON, return None if file doesn't exist or is invalid."""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def estimate_time_remaining(elapsed_seconds: float, trials_completed: int, total_trials: int) -> str:
    """Estimate time remaining based on current progress."""
    if trials_completed == 0:
        return "Calculating..."

    avg_time_per_trial = elapsed_seconds / trials_completed
    remaining_trials = total_trials - trials_completed
    remaining_seconds = avg_time_per_trial * remaining_trials

    return str(timedelta(seconds=int(remaining_seconds)))


def format_coefficient(coeff: float) -> str:
    """Format coefficient for display."""
    if coeff == 0:
        return "0.0"
    elif abs(coeff) >= 1:
        return f"{coeff:.4f}"
    elif abs(coeff) >= 0.001:
        return f"{coeff:.6f}"
    else:
        return f"{coeff:.2e}"


def monitor_progress(results_path: str, interval: int, n_trials: int):
    """Monitor optimization progress and display live updates."""
    print(f"🔍 Monitoring optimization progress: {results_path}")
    print(f"   Refresh interval: {interval}s")
    print(f"   Target trials: {n_trials}")
    print()

    start_time = time.time()
    last_trials = 0
    last_best_loss = None

    try:
        while True:
            results = load_results(results_path)

            # Clear screen and print header
            os.system('clear' if os.name != 'nt' else 'cls')
            print("=" * 80)
            print(f"📊 INNER LOOP OPTIMIZATION MONITOR".center(80))
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

            if results is None:
                print("⏳ Waiting for optimization to start...")
                print(f"   Results file: {results_path}")
                print(f"   Status: Not found or empty")
                print()
                print("💡 Tip: Run 'python inner_loop_optimize.py --iteration N --n-trials 50' in another terminal")
            else:
                # Extract metrics
                best_loss = results.get('best_loss', float('inf'))
                best_alpha = results.get('best_alpha', [])
                best_beta = results.get('best_beta', [])
                trials_completed = results.get('trials_completed',
                                               results.get('n_trials',
                                                          len(results.get('error_log', []))))
                converged_early = results.get('converged_early', False)
                num_errors = len(results.get('error_log', []))

                # Time estimates
                elapsed = time.time() - start_time
                eta = estimate_time_remaining(elapsed, trials_completed, n_trials)

                # Progress bar
                progress = min(trials_completed / n_trials, 1.0)
                bar_width = 50
                filled = int(bar_width * progress)
                bar = "█" * filled + "░" * (bar_width - filled)

                print(f"Progress: [{bar}] {trials_completed}/{n_trials} trials ({progress*100:.1f}%)")
                print()

                # Status
                if converged_early:
                    print("✅ Status: CONVERGED EARLY (no improvement in 50 trials)")
                elif trials_completed >= n_trials:
                    print("✅ Status: OPTIMIZATION COMPLETE")
                else:
                    print(f"🔄 Status: RUNNING (Trial {trials_completed + 1}/{n_trials})")

                print()

                # Performance metrics
                print("📈 Performance:")
                print(f"   Best Loss:     {best_loss:.2f}%")

                # Show improvement rate
                if last_best_loss is not None and last_best_loss != best_loss:
                    improvement = last_best_loss - best_loss
                    improvement_pct = (improvement / last_best_loss) * 100
                    arrow = "↓" if improvement > 0 else "↑"
                    print(f"   Change:        {arrow} {abs(improvement):.2f}% ({improvement_pct:+.1f}%)")

                print()

                # Best coefficients
                print("🎯 Best Coefficients:")
                if best_alpha:
                    alpha_str = ", ".join([format_coefficient(a) for a in best_alpha[:3]])
                    print(f"   Alpha: [{alpha_str}]")
                if best_beta:
                    beta_str = ", ".join([format_coefficient(b) for b in best_beta[:5]])
                    if len(best_beta) > 5:
                        beta_str += f", ... ({len(best_beta)} total)"
                    print(f"   Beta:  [{beta_str}]")

                print()

                # Errors
                if num_errors > 0:
                    print(f"⚠️  Errors: {num_errors} trial(s) failed")
                    print()

                # Time estimates
                print("⏱️  Timing:")
                print(f"   Elapsed:       {timedelta(seconds=int(elapsed))}")
                print(f"   ETA:           {eta}")

                if trials_completed > last_trials:
                    trials_since_last = trials_completed - last_trials
                    time_since_last = interval
                    rate = trials_since_last / (time_since_last / 60)  # trials per minute
                    print(f"   Rate:          {rate:.2f} trials/min")

                print()

                # Next action
                if trials_completed >= n_trials or converged_early:
                    print("=" * 80)
                    print("🎉 OPTIMIZATION COMPLETE!")
                    print()
                    print("Next steps:")
                    print(f"  1. Review results: cat {results_path} | jq '.detailed_diagnostics'")
                    print("  2. Analyze hypotheses and extract error patterns (Phase 4)")
                    print("  3. Run CV tests if all hypotheses confirmed (Phase 5)")
                    print()
                    break

                # Update tracking vars
                last_trials = trials_completed
                last_best_loss = best_loss

            print("=" * 80)
            print(f"Press Ctrl+C to exit | Next update in {interval}s")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor inner loop Bayesian optimization progress"
    )
    parser.add_argument(
        '--iteration',
        type=int,
        required=True,
        help='Iteration number (monitors iterations/iter{N}/inner_loop_results.json)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Refresh interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Expected number of trials (default: 50)'
    )

    args = parser.parse_args()

    # Construct path to results file based on iteration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.dirname(script_dir)
    results_path = os.path.join(training_dir, f"iterations/iter{args.iteration}", "inner_loop_results.json")

    monitor_progress(results_path, args.interval, args.n_trials)


if __name__ == '__main__':
    main()
