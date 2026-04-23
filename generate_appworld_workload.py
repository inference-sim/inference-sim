#!/usr/bin/env python3
"""
Generate workload YAML from exgentic appworld trace data.
Reads actual session traces and produces accurate workload specification.
"""

import json
import yaml
import glob
import math
from datetime import datetime
from pathlib import Path
import tiktoken

def analyze_session(session_dir):
    """Analyze a single session directory and extract all workload parameters."""
    session_id = Path(session_dir).name

    # Read session metadata
    with open(f'{session_dir}/session.json') as f:
        session_data = json.load(f)
        task_id = session_data.get('task_id', 'unknown')
        task = session_data['task']
        context_data = session_data.get('context', {})
        actions = session_data.get('actions', [])

    # Read litellm traces from AGENT (not benchmark) - appworld uses agent traces
    agent_trace_file = f'{session_dir}/agent/litellm/trace.jsonl'
    if not Path(agent_trace_file).exists():
        raise FileNotFoundError(f"Agent trace not found: {agent_trace_file}")

    with open(agent_trace_file) as f:
        traces = [json.loads(line) for line in f]

    # Filter to real conversation turns (prompt_tokens >= 1000 to skip warmup)
    # AppWorld has very high token counts due to tool schemas
    real_turns = [t for t in traces if t['prompt_tokens'] >= 10000]

    if len(real_turns) == 0:
        raise ValueError(f"No real turns found in {session_id}")

    # Extract incremental input token statistics
    # BLIS input_distribution samples NEW tokens per round (not cumulative context)
    # Compute: new_input[i] = prompt_tokens[i] - (prompt_tokens[i-1] + completion_tokens[i-1])
    incremental_inputs = []
    for i, turn in enumerate(real_turns):
        if i == 0:
            # First round: all prompt_tokens are new input
            incremental_inputs.append(turn['prompt_tokens'])
        else:
            # Subsequent rounds: subtract previous cumulative context
            prev_cumulative = real_turns[i-1]['prompt_tokens'] + real_turns[i-1]['completion_tokens']
            new_input = turn['prompt_tokens'] - prev_cumulative
            incremental_inputs.append(max(1, new_input))  # Guard against negative (trace anomalies)

    input_mean = sum(incremental_inputs) / len(incremental_inputs)
    input_variance = sum((x - input_mean)**2 for x in incremental_inputs) / len(incremental_inputs)
    input_std = math.sqrt(input_variance)

    # Extract output token statistics
    output_tokens = [t['completion_tokens'] for t in real_turns]
    output_mean = sum(output_tokens) / len(output_tokens)
    output_variance = sum((x - output_mean)**2 for x in output_tokens) / len(output_tokens)
    output_std = math.sqrt(output_variance)
    output_cv = output_std / output_mean if output_mean > 0 else 0

    # Compute prefix length: first message + tool schemas
    # First message (task + context)
    ctx = "".join(f"\n<{k}>\n{v}\n</{k}>" for k, v in context_data.items())
    first_message_content = f"{task}\n{ctx}"

    # Tool schemas (passed via tools= parameter in litellm API)
    # Approximate token count for actions JSON
    actions_json = json.dumps(actions)

    # Tokenize using tiktoken (GPT-4 tokenizer)
    enc = tiktoken.get_encoding('cl100k_base')
    message_tokens = len(enc.encode(first_message_content))
    tools_tokens = len(enc.encode(actions_json))
    prefix_length = message_tokens + tools_tokens

    # Determine output distribution type based on CV (coefficient of variation)
    # CV < 0.6: Gaussian (consistent responses)
    # CV > 1.2: Exponential (heavy-tailed, rare long responses)
    # 0.6 <= CV <= 1.2: Use Gaussian (border cases)
    if output_cv > 1.2:
        output_dist_type = 'exponential'
    else:
        output_dist_type = 'gaussian'

    # Calculate inter-request times (time from request N to request N+1)
    timestamps_us = []
    for t in real_turns:
        ts_str = t['timestamp']
        dt = datetime.fromisoformat(ts_str.replace('+00:00', '+0000'))
        timestamps_us.append(int(dt.timestamp() * 1e6))

    inter_request_times = []
    for i in range(len(timestamps_us) - 1):
        inter_request_times.append(timestamps_us[i+1] - timestamps_us[i])

    avg_inter_request_us = int(sum(inter_request_times) / len(inter_request_times)) if inter_request_times else 0

    # Timing window (first request to last request)
    window_start_us = timestamps_us[0] - timestamps_us[0]  # Relative to first
    window_end_us = timestamps_us[-1] - timestamps_us[0]

    # Number of rounds
    num_rounds = len(real_turns)

    return {
        'session_id': session_id,
        'task_id': task_id,
        'num_rounds': num_rounds,
        'input_mean': int(round(input_mean)),
        'input_std': int(round(input_std)),
        'input_min': min(incremental_inputs),
        'input_max': max(incremental_inputs),
        'output_mean': int(round(output_mean)),
        'output_std': int(round(output_std)),
        'output_min': min(output_tokens),
        'output_max': max(output_tokens),
        'output_cv': output_cv,
        'output_dist_type': output_dist_type,
        'avg_inter_request_us': avg_inter_request_us,
        'window_start_us': window_start_us,
        'window_end_us': window_end_us,
        'first_timestamp_us': timestamps_us[0],
        'prefix_length': prefix_length,
    }

def generate_workload_yaml(sessions_data, base_timestamp_us):
    """Generate workload YAML structure from analyzed session data."""

    # Sort sessions chronologically
    sessions_data.sort(key=lambda s: s['first_timestamp_us'])

    # Calculate relative timing windows
    for session in sessions_data:
        session['relative_start_us'] = session['first_timestamp_us'] - base_timestamp_us
        session['relative_end_us'] = session['relative_start_us'] + session['window_end_us']

    # Build YAML structure
    # Note: horizon is omitted - BLIS defaults to math.MaxInt64 (infinite)
    workload = {
        'version': '2',
        'seed': 400,
        'category': 'agentic-tool-use',
        'aggregate_rate': 1.0,
    }

    # Add header comment
    header_comment = """
# AppWorld agentic tool-use workload
# Generated from exgentic appworld trace data
#
# IMPORTANT NOTES:
# - input_distribution samples INCREMENTAL new input per round (not cumulative context)
# - context_growth: accumulate → BLIS automatically builds cumulative context internally
# - think_time_us values are INTER-REQUEST TIMES (includes tool execution + inference + think time)
# - prefix_length: Cacheable system prompt + task + tool API schemas (separate from per-round input)
# - Output distributions vary by session (chosen by coefficient of variation):
#   - Gaussian (CV<0.6): Consistent response lengths
#   - Exponential (CV>1.2): Heavy-tailed with rare long tool output dumps
"""

    clients = []
    for i, session in enumerate(sessions_data, 1):
        # Build output distribution based on type
        if session['output_dist_type'] == 'exponential':
            output_dist = {
                'type': 'exponential',
                'params': {
                    'mean': session['output_mean'],
                }
            }
        else:
            output_dist = {
                'type': 'gaussian',
                'params': {
                    'mean': session['output_mean'],
                    'std_dev': session['output_std'],
                    'min': session['output_min'],
                    'max': session['output_max'],
                }
            }

        client = {
            'id': f'appworld-session-{i}',
            'tenant_id': 'appworld-benchmark',
            'slo_class': 'standard',
            'rate_fraction': round(1.0 / len(sessions_data), 3),
            'streaming': True,
            'arrival': {'process': 'poisson'},
            'input_distribution': {
                'type': 'gaussian',
                'params': {
                    'mean': session['input_mean'],
                    'std_dev': session['input_std'],
                    'min': session['input_min'],
                    'max': session['input_max'],
                }
            },
            'output_distribution': output_dist,
            'prefix_length': session['prefix_length'],
            'reasoning': {
                'multi_turn': {
                    'max_rounds': session['num_rounds'],
                    'think_time_us': session['avg_inter_request_us'],
                    'context_growth': 'accumulate',
                }
            },
            'closed_loop': True,
            'timeout': 300000000,
            'lifecycle': {
                'windows': [{
                    'start_us': session['relative_start_us'],
                    'end_us': session['relative_end_us'],
                }]
            }
        }

        clients.append(client)

    workload['clients'] = clients

    return workload, header_comment

def main():
    # Path to appworld sessions
    sessions_dir = Path(__file__).parent / 'exgentic/outputs/appworld_test_normal_10/sessions'
    output_file = Path(__file__).parent / 'workload_exgentic_appworld.yaml'

    if not sessions_dir.exists():
        print(f"ERROR: Sessions directory not found: {sessions_dir}")
        return 1

    print("Analyzing appworld sessions...")
    sessions_data = []

    for session_dir in sorted(sessions_dir.glob('*/')):
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name

        try:
            session_info = analyze_session(session_dir)
            sessions_data.append(session_info)
            print(f"  ✓ Session {session_info['session_id']} (Task {session_info['task_id']}): "
                  f"{session_info['num_rounds']} rounds, "
                  f"input={session_info['input_mean']}±{session_info['input_std']}, "
                  f"output={session_info['output_mean']}±{session_info['output_std']} "
                  f"(CV={session_info['output_cv']:.2f}, {session_info['output_dist_type']})")
        except Exception as e:
            print(f"  ✗ Failed to analyze {session_dir.name}: {e}")
            continue

    if not sessions_data:
        print("ERROR: No sessions found!")
        return 1

    # Get base timestamp from first session
    base_timestamp_us = min(s['first_timestamp_us'] for s in sessions_data)

    print(f"\nGenerating workload YAML with {len(sessions_data)} sessions...")
    workload, header_comment = generate_workload_yaml(sessions_data, base_timestamp_us)

    # Write YAML
    print(f"Writing to {output_file}...")

    with open(output_file, 'w') as f:
        # Write header comment
        for line in header_comment.strip().split('\n'):
            f.write(f"{line}\n")
        f.write("\n")

        # Write YAML
        yaml_str = yaml.dump(workload, default_flow_style=False, sort_keys=False, allow_unicode=True)
        f.write(yaml_str)

    max_end = max(s['relative_end_us'] for s in sessions_data)

    print(f"✓ Successfully generated {output_file}")
    print(f"\nSummary:")
    print(f"  Sessions: {len(sessions_data)}")
    print(f"  Horizon: omitted (BLIS default: infinite)")
    print(f"  Last session ends: {max_end/1e6:.1f}s")
    print(f"  Round range: {min(s['num_rounds'] for s in sessions_data)}-{max(s['num_rounds'] for s in sessions_data)}")
    print(f"  Output distributions: {sum(1 for s in sessions_data if s['output_dist_type']=='gaussian')} gaussian, "
          f"{sum(1 for s in sessions_data if s['output_dist_type']=='exponential')} exponential")

    return 0

if __name__ == '__main__':
    exit(main())
