# Feature Specification: Phase 1A — Infrastructure: Nodes, GPUs, Instances

**Feature Branch**: `001-infra-nodes-gpus-instances`
**Created**: 2026-03-13
**Status**: Draft
**Source**: [Discussion #402 comment v4 (I-1–I-5)](https://github.com/inference-sim/inference-sim/discussions/402#discussioncomment-15901661)

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Configure a Node-Pool Infrastructure Topology (Priority: P1)

A researcher configuring a multi-GPU cluster simulation defines one or more node pools,
each specifying a GPU type (e.g., H100), GPUs per node, and initial/min/max node counts.
The simulator materializes concrete nodes and GPUs with unique IDs, and tracks GPU
inventory per node.

**Why this priority**: Everything else in Phase 1A and autoscaling depends on nodes and
GPUs existing as first-class addressable entities. Without identifiable hardware,
placement, lifecycle, and routing cannot be specified.

**Independent Test**: A simulation configured with two node pools (H100 ×8 and A100 ×4)
produces a cluster with the correct node count, each node carrying the declared number of
named GPUs of the right type.

**Acceptance Scenarios**:

1. **Given** a `node_pools` config with `h100-pool` (8 H100 GPUs/node, 2 initial nodes),
   **When** the simulation initializes,
   **Then** 2 nodes exist each carrying 8 distinct GPU IDs of type H100 — 16 GPUs total.

2. **Given** a node pool with `initial_nodes: 0`,
   **When** the simulation initializes,
   **Then** the pool exists with zero nodes and zero GPUs; the pool's capacity limits
   (min/max) are respected throughout the run.

3. **Given** a multi-pool config (one H100 pool and one A100 pool),
   **When** the simulation initializes,
   **Then** GPUs across pools are never confused — each GPU ID encodes its pool and node
   of origin.

---

### User Story 2 — Place Instances onto Nodes Using Bin-Packing (Priority: P1)

When an instance (model server) is created, the simulator assigns it to a node that has
enough free GPUs of the required type (determined by the instance's TP degree). If no
existing node has capacity, the instance enters a **pending** state.

**Why this priority**: Instance placement is the mechanism through which model autoscaling
(Phase 1C) interacts with cluster autoscaling. Pending instances are the signal that
triggers node provisioning.

**Independent Test**: Requesting 3 instances of TP=4 on a single 8-GPU node results in
2 instances placed and 1 instance pending; the first node shows all 8 GPUs allocated.

**Acceptance Scenarios**:

1. **Given** a node with 8 H100 GPUs and two TP=4 instance requests for the same GPU type,
   **When** the instance scheduler runs,
   **Then** both instances are placed (consuming GPUs 0–3 and 4–7 respectively), and the
   node shows 0 free GPUs.

2. **Given** a node with 8 H100 GPUs fully allocated and a new TP=4 instance request,
   **When** the instance scheduler runs,
   **Then** the instance enters pending state — it is not placed on any node.

3. **Given** an instance is terminated,
   **When** the placement accounting updates,
   **Then** the GPUs that instance occupied are immediately returned to the node's free
   pool and available for new placements.

4. **Given** two instance requests with different GPU type requirements (H100 vs A100),
   **When** the instance scheduler runs,
   **Then** each instance is placed only on a node in a pool with the matching GPU type.

---

### User Story 3 — Model Node Provisioning and Termination Lifecycle (Priority: P2)

Nodes transition through defined lifecycle states with configurable delays. A node that is
provisioning is not yet available for instance placement. Cost accrues only while a node
exists (Provisioning through Draining inclusive).

**Why this priority**: Realistic provisioning delay (30 s – 5 min) is the key
differentiator between "instance on existing node" (fast) and "need a new node first"
(slow), which is central to every autoscaling experiment.

**Independent Test**: Adding a node with a 120 s provisioning delay means instances
cannot be placed on it for 120 simulation seconds; cost starts when provisioning begins
and stops when the node reaches Terminated.

**Acceptance Scenarios**:

1. **Given** a node pool with a 120 s provisioning distribution,
   **When** a node is provisioned at time T,
   **Then** the node transitions to `Ready` at T+120 s (±distribution variance); pending
   instances can only be placed after the `Ready` transition.

2. **Given** a `Ready` node with no instances,
   **When** a drain command is issued,
   **Then** the node transitions immediately to `Draining` then `Terminated` (no instances
   to wait for), and its GPUs are removed from the free pool.

3. **Given** a `Draining` node with active instances,
   **When** all instances on that node reach `Terminated`,
   **Then** the node transitions to `Terminated` and cost accounting stops.

---

### User Story 4 — Model Instance Startup Phases Including Warm-Up (Priority: P2)

Instances transition through Loading → WarmingUp → Active, with configurable per-phase
delays. Routing skips non-Active instances. The warm-up penalty (cold KV cache) is
visible in TTFT for the first N requests served by a newly active instance.

**Why this priority**: The warm-up cost is the key variable in H-WarmupCost experiments
and in understanding whether autoscaling buys latency improvement quickly or slowly.

**Independent Test**: A newly loaded instance receiving its first request shows an
elevated TTFT compared to a fully warmed instance; after N warm-up requests the
instance's TTFT matches the baseline.

**Acceptance Scenarios**:

1. **Given** an instance with a 60 s loading delay,
   **When** the instance is scheduled at time T,
   **Then** no requests are routed to it until T+60 s (it is not in `Active` state).

2. **Given** a newly `Active` instance receiving its first request,
   **When** latency is computed,
   **Then** TTFT reflects the KV cold-start penalty (higher than a warm instance).

3. **Given** an instance in `Draining` state with the `WAIT` drain policy,
   **When** a new request arrives for that model,
   **Then** the draining instance is excluded from routing; only `Active` instances
   receive the new request.

4. **Given** an instance in `Draining` state with the `REDIRECT` drain policy,
   **When** the drain begins,
   **Then** queued requests on that instance are migrated to other `Active` instances
   of the same model.

---

### User Story 5 — Route Requests to Active Instances of the Correct Model (Priority: P2)

Requests carry a target model identifier. The router considers only `Active` instances of
that model. Per-model metrics (TTFT, E2E latency, throughput) appear independently in the
output.

**Why this priority**: Multi-model routing is the prerequisite for validating H-MultiModel
and for the two-level autoscaler to operate on model-specific signals.

**Independent Test**: A cluster running two models (llama, qwen) where all llama instances
are draining causes llama requests to queue while qwen requests are served normally;
separate per-model metrics are present in the output JSON.

**Acceptance Scenarios**:

1. **Given** a cluster with 2 llama instances and 2 qwen instances,
   **When** a llama request arrives,
   **Then** it is routed only to a llama instance; qwen instances are never considered.

2. **Given** all instances of model M are non-Active,
   **When** a request for model M arrives,
   **Then** the request waits (or is rejected by admission policy) — it is never sent to
   an instance of a different model.

3. **Given** a completed simulation run with 2 models,
   **When** metrics are output,
   **Then** TTFT p99, E2E p99, and throughput appear separately for each model name.

---

### Edge Cases

- What happens when a node pool's `max_nodes` is reached and instances are still pending?
- How does placement behave when TP degree exceeds GPUs-per-node in all available pools?
- What happens if a node enters `Draining` while an instance on it is still in `Loading`?
- How does the GPU allocation conservation invariant hold across concurrent placement
  and termination events in the same simulation step?
- What happens when two autoscaler decisions simultaneously request placement on the
  same node's last available GPUs?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The simulator MUST support configuring one or more node pools, each with a
  GPU type, GPUs-per-node count, GPU memory, and initial/min/max node counts.

- **FR-002**: Every node and GPU MUST have a unique identifier; GPU IDs MUST be
  traceable to their parent node and node pool.

- **FR-003**: The instance scheduler MUST place an instance on a node that has at least
  `tp_degree` free GPUs of the correct type; if no such node exists, the instance MUST
  enter a pending state.

- **FR-004**: The simulator MUST track free vs. allocated GPU inventory per node at all
  times, maintaining the invariant `allocated_gpus + free_gpus = total_gpus` per node.

- **FR-005**: Nodes MUST support four lifecycle states — `Provisioning`, `Ready`,
  `Draining`, `Terminated` — and transitions MUST follow that valid sequence.

- **FR-006**: Node provisioning delay MUST be configurable via a distribution (to model
  VM spin-up variance); cost MUST accrue from state entry through `Terminated`.

- **FR-007**: Instances MUST support six lifecycle states — `Scheduling`, `Loading`,
  `WarmingUp`, `Active`, `Draining`, `Terminated` — and only `Active` instances MUST
  receive new requests.

- **FR-008**: Instance loading delay and warm-up request count MUST be configurable; the
  `WarmingUp` phase MUST apply a KV cold-start penalty to TTFT.

- **FR-009**: Three drain policies MUST be supported: `IMMEDIATE` (kill in-flight),
  `WAIT` (finish in-flight, reject new), `REDIRECT` (migrate queued to other instances).

- **FR-010**: Each request MUST carry a target model identifier; routing MUST consider
  only `Active` instances of the specified model.

- **FR-011**: Per-model metrics MUST appear in the JSON output: TTFT (p50/p99), E2E
  latency (p50/p99), and throughput, broken out by model name.

- **FR-012**: Node pool configuration MUST be expressible in YAML with strict field
  parsing; unknown fields MUST cause parse errors.

- **FR-013**: GPU allocation and release MUST be transactional — if placement fails
  mid-loop, all partial GPU assignments MUST be rolled back.

- **FR-014**: The simulation MUST remain deterministic: same seed and same node pool
  configuration MUST produce byte-identical stdout, including placement decisions and
  lifecycle timings.

### Key Entities

- **NodePool**: Logical group with a name, GPU type, GPUs-per-node, GPU memory, and
  initial/min/max node count. Analogous to a Karpenter NodeClass.

- **Node**: A machine with a unique ID, belonging to one NodePool, carrying a fixed
  inventory of GPUs. States: `Provisioning`, `Ready`, `Draining`, `Terminated`.

- **GPU**: A single accelerator device with a unique ID, a type, memory capacity, and
  a reference to its parent node. State: free or allocated to one instance.

- **Instance** (model server): A model-serving process with a unique ID, a target model
  name, a TP degree (GPU count), a GPU type requirement, and a lifecycle state. Occupies
  specific GPU IDs on one node when placed.

- **PlacementRecord**: Trace record capturing instance-to-node-to-GPU binding decisions,
  including timestamp and outcome (placed / pending / evicted).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A simulation with N node pools correctly materializes the declared number
  of nodes and GPUs; the GPU conservation invariant
  (`allocated + free = total`) holds throughout every run with zero violations.

- **SC-002**: Instance placement never violates GPU type constraints — zero type
  mismatches across all placement decisions in the test suite.

- **SC-003**: The simulated time gap between "instance on an existing ready node" and
  "instance requiring a new node to provision" matches the configured provisioning delay
  distribution within statistical tolerance across 100+ simulation runs.

- **SC-004**: The warm-up TTFT penalty is measurable: the first N requests to a newly
  active instance show higher TTFT p99 than a fully warmed instance serving the same
  request profile.

- **SC-005**: In a multi-model cluster, per-model TTFT p99 and throughput appear
  independently in the output JSON; aggregate metrics are consistent with per-model sums.

- **SC-006**: All tests added for Phase 1A pass within the 60-second budget; every
  golden test has a companion GPU-conservation invariant test.

- **SC-007**: Simulation output is byte-identical across 5 repeated runs with the same
  seed and node-pool configuration, confirming determinism is preserved.

## Assumptions

- Node pool configuration is provided at simulation startup; nodes are not dynamically
  added mid-run except via autoscaler events (Phase 1C). Autoscaler interfaces are out
  of scope for this phase.
- TP degree is the sole GPU-count requirement per instance; NUMA or PCIe topology
  constraints are not modeled.
- GPU memory capacity is tracked for KV auto-calculation purposes; GPU memory
  fragmentation is not modeled.
- The warm-up penalty is an additive TTFT overhead for the first N requests; the exact
  warm-up model (fixed N, exponential decay, etc.) is left to the implementation plan.
- Cost is modeled as `node_seconds` (a cloud-bill proxy); per-GPU-type pricing is a
  Phase 1C concern.
