#!/usr/bin/env python3
"""
Test convergence callback for inner loop optimizer.

This test verifies that early stopping works correctly when
the best loss plateaus (≤1% improvement in 50 trials).
"""

import optuna
from optuna.trial import Trial


def test_convergence_callback():
    """Test that convergence callback stops optimization correctly."""

    # Track whether callback stopped the study
    converged_early = [False]

    def convergence_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Stop if best loss hasn't improved >1% in last 50 trials."""
        n = len(study.trials)
        if n <= 50:
            return  # Need more than 50 trials to check 50-trial window

        # Get best loss from all trials up to 50 trials ago (trials 0 to n-51)
        trials_before_window = study.trials[:n-50]
        best_loss_50_ago = min(t.value for t in trials_before_window if t.value is not None)

        # Get current best loss
        current_best = study.best_value

        # Calculate improvement
        improvement = (best_loss_50_ago - current_best) / best_loss_50_ago

        if improvement <= 0.01:  # ≤1% improvement
            print(f"\n[Test] Stopping at trial {n}")
            print(f"[Test] Best loss 50 trials ago: {best_loss_50_ago:.6f}")
            print(f"[Test] Current best loss: {current_best:.6f}")
            print(f"[Test] Improvement: {improvement*100:.2f}% (threshold: >1.00%)")
            converged_early[0] = True
            study.stop()

    # Objective function that plateaus after 60 trials
    # Loss decreases quickly, then plateaus
    def objective(trial: Trial) -> float:
        trial_number = trial.number
        if trial_number < 10:
            return 100.0 - trial_number * 5  # Fast improvement: 100 → 50
        elif trial_number < 60:
            return 50.0 - (trial_number - 10) * 0.5  # Moderate: 50 → 25
        else:
            # Plateau: oscillate around 25 with <1% changes
            import random
            return 25.0 + random.uniform(-0.1, 0.1)

    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=150,  # Request 150 trials
        callbacks=[convergence_callback],
        show_progress_bar=False
    )

    # Verify early stopping occurred
    actual_trials = len(study.trials)
    print(f"\n[Test] Requested trials: 150")
    print(f"[Test] Actual trials: {actual_trials}")
    print(f"[Test] Converged early: {converged_early[0]}")
    print(f"[Test] Final best loss: {study.best_value:.6f}")

    # Assertions
    assert converged_early[0], "Expected early convergence but ran all trials"
    assert actual_trials < 150, f"Expected <150 trials, got {actual_trials}"
    assert actual_trials >= 60, f"Stopped too early at trial {actual_trials}"
    assert 110 <= actual_trials <= 120, f"Expected ~110 trials (60 + 50 window), got {actual_trials}"

    print("\n✅ Convergence test passed!")


if __name__ == "__main__":
    test_convergence_callback()
