# Evaluation Experiment Overview

## Experimental Dimensions

The evaluation will systematically vary parameters across four key dimensions:

**1. Hardware Configurations (3 variants)**
- Different GPU architectures with varying memory and compute capabilities
- H100, A100-80, L40S
- Native vs. fallback operation support impacts quantization performance

**2. Model Architectures (8 variants)**
- Dense and sparse mixture-of-experts topologies
- Parameter scales ranging from 7B to 141B

**3. Configuration Parameters (5 knobs)**
- max-num-batched-tokens variation
- Memory offloading: enabled/disabled
- GPU memory utilization: two threshold settings
- Tensor parallelism: varies by model requirements
- Expert parallelism: for MoE models only

**4. Workload Profiles (4 types)**
- Varied input/output token distributions
- Realistic workloads with multiple stages and multi-turn chat
- Varying request rates through stages
