// Package sim provides the core discrete-event simulation engine for BLIS.
//
// # Reading Guide
//
// Start with these three files to understand the simulation kernel:
//   - request.go: Request lifecycle (queued → running → completed) and state machine
//   - event.go: Event types that drive the simulation (Arrival, Step, Scheduled, etc.)
//   - simulator.go: The event loop, batch formation, and step execution
//
// # Architecture
//
// The sim package defines interfaces and bridge types; implementations live in
// sub-packages:
//   - sim/kv/: KV cache implementations (single-tier GPU, tiered GPU+CPU)
//   - sim/latency/: Latency models (blackbox alpha/beta, roofline FLOPs/bandwidth)
//   - sim/cluster/: Multi-instance cluster orchestration
//   - sim/workload/: Workload generation and trace replay
//   - sim/trace/: Decision trace recording
//
// Sub-packages register their implementations via init() functions that set
// package-level factory variables (NewLatencyModelFunc, NewKVStoreFromConfig).
//
// # Key Interfaces
//
// The extension points are single-method or small interfaces:
//   - LatencyModel: step time, queueing time, scheduling/preemption/output processing overheads
//   - KVStore: block allocation, eviction, prefix caching, capacity queries
//   - RoutingPolicy: select target instance given cluster snapshots
//   - AdmissionPolicy: accept or reject incoming requests
//   - InstanceScheduler: order requests within a single instance's wait queue
//   - PriorityPolicy: compute priority scores for scheduling
//   - BatchFormation: form batches from waiting requests with KV constraints
//
// See docs/extension-recipes.md for step-by-step guides to extend each interface.
package sim
