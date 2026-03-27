#!/usr/bin/env python3
"""
Pre-flight validation for agentic latency backend integration.

Checks that the backend is properly registered and integrated before
starting expensive Bayesian optimization. Catches common integration
issues early with actionable error messages.
"""

import os
import sys
import subprocess
import re
import yaml


def validate_backend_integration(backend_name: str, project_root: str, iteration: int = None) -> bool:
    """
    Validate that backend is fully integrated into BLIS.

    Args:
        backend_name: Name of the latency backend to validate
        project_root: Path to the project root directory
        iteration: Optional iteration number to load coefficient bounds from

    Returns True if all checks pass, False otherwise.
    """
    errors = []
    warnings = []

    print(f"🔍 Validating backend '{backend_name}' integration...")
    print()

    # 1. Check sim/bundle.go registration
    bundle_path = os.path.join(project_root, 'sim', 'bundle.go')
    try:
        with open(bundle_path) as f:
            content = f.read()
            # Check for backend in validLatencyBackends map
            if f'"{backend_name}"' not in content:
                errors.append(
                    f"Backend '{backend_name}' not found in sim/bundle.go validLatencyBackends map.\n"
                    f"  → Add: validLatencyBackends = map[string]bool{{..., \"{backend_name}\": true}}"
                )
            else:
                print(f"  ✅ Backend registered in sim/bundle.go")
    except FileNotFoundError:
        errors.append(f"File not found: {bundle_path}")

    # 2. Check cmd/root.go auto-fetch block
    root_cmd_path = os.path.join(project_root, 'cmd', 'root.go')
    try:
        with open(root_cmd_path) as f:
            content = f.read()

            # Check for auto-fetch block (like roofline/trained-roofline)
            if f'backend == "{backend_name}"' not in content:
                errors.append(
                    f"No auto-fetch block for '{backend_name}' in cmd/root.go.\n"
                    f"  → Add auto-fetch block similar to 'trained-roofline' (lines 555-627)"
                )
            else:
                print(f"  ✅ Auto-fetch block found in cmd/root.go")

            # Check analytical backends condition
            # Look for the analytical backends block that has roofline AND crossmodel
            analytical_pattern = r'if backend == "roofline" \|\| backend == "crossmodel".*?\{'
            match = re.search(analytical_pattern, content, re.DOTALL)
            if match:
                block_start = match.start()
                # Extract the full condition line (everything up to the opening brace)
                line_end = content.find('{', block_start)
                if line_end != -1:
                    block_content = content[block_start:line_end]
                    if f'backend == "{backend_name}"' not in block_content:
                        errors.append(
                            f"Backend '{backend_name}' not in analytical backends condition in cmd/root.go.\n"
                            f"  → Add to condition around line 739: || backend == \"{backend_name}\""
                        )
                    else:
                        print(f"  ✅ Backend in analytical backends config loading condition")
    except FileNotFoundError:
        errors.append(f"File not found: {root_cmd_path}")

    # 3. Check BLIS binary exists and is recent
    blis_binary = os.path.join(project_root, 'blis')
    if not os.path.exists(blis_binary):
        errors.append(
            f"BLIS binary not found at {blis_binary}.\n"
            f"  → Run: go build -o blis main.go"
        )
    else:
        # Check if binary is newer than evolved_model.go
        evolved_path = os.path.join(project_root, 'sim', 'latency', 'evolved_model.go')
        if os.path.exists(evolved_path):
            blis_mtime = os.path.getmtime(blis_binary)
            evolved_mtime = os.path.getmtime(evolved_path)
            if evolved_mtime > blis_mtime:
                warnings.append(
                    f"BLIS binary is older than evolved_model.go.\n"
                    f"  → Recompile: go build -o blis main.go"
                )
            else:
                print(f"  ✅ BLIS binary is up-to-date")
        else:
            print(f"  ✅ BLIS binary exists")

    # 4. Test run with backend
    print(f"\n  🧪 Running test simulation with backend '{backend_name}'...")

    # Load coefficient bounds from iteration
    alpha_coeffs = None
    beta_coeffs = None

    if iteration is not None:
        training_dir = os.path.join(project_root, 'training')
        bounds_path = os.path.join(training_dir, f'iterations/iter{iteration}', 'coefficient_bounds.yaml')
        if os.path.exists(bounds_path):
            try:
                with open(bounds_path) as f:
                    bounds = yaml.safe_load(f)
                    alpha_initial = bounds.get('alpha_initial', [])
                    beta_initial = bounds.get('beta_initial', [])
                    if alpha_initial and beta_initial:
                        alpha_coeffs = ','.join(str(x) for x in alpha_initial)
                        beta_coeffs = ','.join(str(x) for x in beta_initial)
                        print(f"  📋 Using coefficients from iteration {iteration}:")
                        print(f"     Alpha: {len(alpha_initial)} coefficients")
                        print(f"     Beta: {len(beta_initial)} coefficients")
                    else:
                        errors.append(
                            f"Coefficient bounds file {bounds_path} missing alpha_initial or beta_initial.\n"
                            f"  → Both alpha_initial and beta_initial are required in coefficient_bounds.yaml"
                        )
            except Exception as e:
                errors.append(
                    f"Failed to load coefficient bounds from {bounds_path}: {e}\n"
                    f"  → Check that the YAML file is valid and contains alpha_initial/beta_initial"
                )
        else:
            errors.append(
                f"Coefficient bounds file not found at {bounds_path}\n"
                f"  → Run Agent 1 to generate coefficient_bounds.yaml for iteration {iteration}"
            )
    else:
        # No iteration specified - use iter0-compatible defaults (3 beta terms)
        alpha_coeffs = '0.0002,0.000001,0.000002'
        beta_coeffs = '1.0,1.0,0.0001'
        warnings.append(
            f"No --iteration flag provided, using iter0-compatible defaults (3 beta terms).\n"
            f"  → For iteration-specific validation, use: python validate_backend.py {backend_name} --iteration N"
        )

    # If we couldn't load coefficients, we can't proceed with the test
    if alpha_coeffs is None or beta_coeffs is None:
        print()
        print("=" * 70)
        print("❌ VALIDATION FAILED:")
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. {err}")
        print()
        print("=" * 70)
        print("\nFix these issues before running optimization.")
        return False

    test_cmd = [
        blis_binary, 'run',
        '--model', 'qwen/qwen3-14b',
        '--latency-model', backend_name,
        '--alpha-coeffs', alpha_coeffs,
        '--beta-coeffs', beta_coeffs,
        '--num-requests', '5'
    ]

    try:
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            timeout=30,
            cwd=project_root
        )

        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='replace')
            errors.append(
                f"Test run failed with backend '{backend_name}'.\n"
                f"  Command: {' '.join(test_cmd)}\n"
                f"  Error: {stderr[:500]}"
            )
        else:
            print(f"  ✅ Test simulation succeeded")
    except subprocess.TimeoutExpired:
        errors.append(f"Test run timed out (>30s) - possible infinite loop or deadlock")
    except Exception as e:
        errors.append(f"Test run crashed: {e}")

    # Print results
    print()
    print("=" * 70)

    if warnings:
        print("⚠️  WARNINGS:")
        for warn in warnings:
            print(f"\n{warn}")
        print()

    if errors:
        print("❌ VALIDATION FAILED:")
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. {err}")
        print()
        print("=" * 70)
        print("\nFix these issues before running optimization.")
        return False
    else:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 70)
        print(f"\nBackend '{backend_name}' is properly integrated and ready for optimization!")
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_backend.py <backend_name> [--iteration N]")
        print()
        print("Examples:")
        print("  python validate_backend.py evolved --iteration 0  # Validate iter0")
        print("  python validate_backend.py evolved --iteration 1  # Validate iter1")
        print()
        print("Note: --iteration flag is REQUIRED for iteration-specific coefficient counts.")
        print("      Without it, defaults to iter0-compatible (3 beta terms).")
        sys.exit(1)

    backend_name = sys.argv[1]
    iteration = None

    # Parse optional --iteration flag
    if len(sys.argv) >= 4 and sys.argv[2] == '--iteration':
        try:
            iteration = int(sys.argv[3])
        except ValueError:
            print(f"Error: Invalid iteration number '{sys.argv[3]}'")
            sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    success = validate_backend_integration(backend_name, project_root, iteration)
    sys.exit(0 if success else 1)
