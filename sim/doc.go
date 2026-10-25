// Package sim is the host of BLIS's per-instance discrete-event simulation: the clock, the
// event queue, when an engine steps and when it parks, the data-parallel lockstep group, the
// progress hook, and the run's report (Metrics).
//
// # Reading Guide
//
//   - simulator.go: the Simulator (the host) and its event loop
//   - event.go: the event types and their priorities
//   - idle_park.go, dp_lockstep.go: parking an idle engine; stepping a DP replica in lockstep
//   - metrics.go, recorder.go: the report the engine writes through its Recorder
//
// # Architecture
//
// The engine -- batch formation, schedulers, the request record, the KV store, LoRA and the
// blis-latency-kernel adapter -- lives in blis-vllm's compatibility engine (compat/engine,
// compat/kv, compat/lora, compat/kernelmodel; inference-sim#2021). The host reaches it through
// one seam, engine_aliases.go, which gives the engine's types their sim names (sim.Request,
// sim.KVStore, sim.LatencyModel, ...). The cluster, the workload and the trace stay here:
//   - sim/cluster/: multi-instance cluster orchestration
//   - sim/workload/: workload generation and trace replay
//   - sim/trace/: decision trace recording
//   - sim/llmd/: the embedded llm-d router
//
// The engine's KV store and adapter registry register their constructors when blis-vllm's
// compat/kv and compat/lora are linked (NewKVStoreFromConfig); inference-sim links them at its
// one build site, cmd/engine_build.go. The latency model is not registered: the caller builds
// it and passes it in (SimConfig.LatencyModel).
package sim
