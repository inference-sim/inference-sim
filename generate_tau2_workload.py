#!/usr/bin/env python3
"""
Generate workload YAML from exgentic tau2 trace data.
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
        task_id = session_data['task_id']
        task = session_data['task']

    # Read context (contains system prompt/policy)
    with open(f'{session_dir}/benchmark/context.json') as f:
        context_data = json.load(f)

    # Read litellm traces
    with open(f'{session_dir}/benchmark/litellm/trace.jsonl') as f:
        traces = [json.loads(line) for line in f]

    # Use all turns from traces
    real_turns = traces

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

    # Compute prefix length by tokenizing the first message (task + context)
    # This replicates what litellm_tool_calling_agent does in start():
    # self._add_message(ChatCompletionUserMessage(role="user", content=f"{self.task}\n{ctx}"))
    ctx = "".join(f"\n<{k}>\n{v}\n</{k}>" for k, v in context_data.items())
    first_message_content = f"{task}\n{ctx}"

    # Tokenize using tiktoken (GPT-4 tokenizer)
    enc = tiktoken.get_encoding('cl100k_base')
    prefix_length = len(enc.encode(first_message_content))

    # Calculate inter-request times (time from request N to request N+1)
    # NOTE: This includes both inference latency and actual user think time
    # We cannot separate them from the available trace data
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
        'seed': 300,
        'category': 'reasoning',
        'aggregate_rate': 1.0,
    }

    # Add header comment
    header_comment = """
# Exact replay of exgentic tau2_retail_10 benchmark execution
# Generated from actual trace data
#
# IMPORTANT NOTES:
# - input_distribution samples INCREMENTAL new input per round (not cumulative context)
# - context_growth: accumulate → BLIS automatically builds cumulative context internally
# - think_time_us values are INTER-REQUEST TIMES (includes inference + actual think time)
# - prefix_length: Cacheable system prompt + task (separate from per-round input)
"""

    clients = []
    for i, session in enumerate(sessions_data, 1):
        client = {
            'id': f'tau2-session-{i}',
            'tenant_id': 'retail-benchmark',
            'slo_class': 'standard',
            'rate_fraction': 0.1,
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
            'output_distribution': {
                'type': 'gaussian',
                'params': {
                    'mean': session['output_mean'],
                    'std_dev': session['output_std'],
                    'min': session['output_min'],
                    'max': session['output_max'],
                }
            },
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

        # Add comments
        client['input_distribution']['params']['_comment'] = \
            f"Task {session['task_id']}: {session['num_rounds']} rounds"
        client['output_distribution']['params']['_comment'] = \
            f"CV={session['output_std']/session['output_mean']:.2f}"
        client['reasoning']['multi_turn']['_comment'] = \
            f"Inter-request time (inference + think): {session['avg_inter_request_us']/1e6:.1f}s"

        clients.append(client)

    workload['clients'] = clients

    return workload, header_comment

def main():
    # Path to tau2 sessions
    sessions_dir = Path(__file__).parent / 'exgentic/outputs/tau2_retail_10/sessions'
    output_file = Path(__file__).parent / 'workload_exgentic_tau2_retail.yaml'

    print("Analyzing tau2 sessions...")
    sessions_data = []

    for session_dir in sorted(sessions_dir.glob('*/')):
        if not session_dir.is_dir():
            continue

        try:
            session_info = analyze_session(session_dir)
            sessions_data.append(session_info)
            print(f"  ✓ Session {session_info['session_id']} (Task {session_info['task_id']}): "
                  f"{session_info['num_rounds']} rounds, "
                  f"input={session_info['input_mean']}±{session_info['input_std']}, "
                  f"output={session_info['output_mean']}±{session_info['output_std']}")
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

        # Write YAML (without Python object tags)
        yaml_str = yaml.dump(workload, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Clean up YAML output (remove comment keys and fix formatting)
        lines = []
        for line in yaml_str.split('\n'):
            if '_comment:' not in line:
                lines.append(line)

        f.write('\n'.join(lines))

    max_end = max(s['relative_end_us'] for s in sessions_data)

    print(f"✓ Successfully generated {output_file}")
    print(f"\nSummary:")
    print(f"  Sessions: {len(sessions_data)}")
    print(f"  Horizon: omitted (BLIS default: infinite)")
    print(f"  Last session ends: {max_end/1e6:.1f}s")
    print(f"  Round range: {min(s['num_rounds'] for s in sessions_data)}-{max(s['num_rounds'] for s in sessions_data)}")

    return 0

if __name__ == '__main__':
    exit(main())
