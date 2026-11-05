package main

// relaxations is release plan §3.3, "Relaxing the deprecation rule until M4": the features
// LIF-7 lets stay past a compatibility-breaking release until the M4 milestone release
// removes them. The release bot's deprecation check skips exactly these. Every other
// deprecated feature, including observe, calibrate, --metrics-path and --results-path, is
// removed in M1.
//
// This list is the plan's, copied because the plan is not a file the bot can read. When the
// plan changes, this list changes in the same pull request; TestRelaxationsMatchThePlan
// pins it, and a reviewer compares the two.
var relaxations = []Relaxation{
	{Repo: "inference-sim", Path: "", Match: "engine_version", Milestone: 4,
		Reason: "the scenario field engine_version (plan §3.3)"},
	{Repo: "blis-vllm", Path: "compat/", Milestone: 4, Reason: "blis-vllm's compat/ (plan §3.3)"},
	{Repo: "blis-vllm", Path: "rules/v0_29/", Milestone: 4, Reason: "rules/v0_29 (plan §3.3)"},
	{Repo: "blis-vllm", Path: "rules/v0_31/", Milestone: 4, Reason: "rules/v0_31 and the v0.31 resolution code (plan §3.3)"},
	{Repo: "blis-substrate", Path: "contracts/rules/", Milestone: 4, Reason: "contracts/rules (plan §3.3)"},
	{Repo: "blis-substrate", Path: "contracts/kernel/enginesettings/", Milestone: 4, Reason: "contracts/kernel/enginesettings (plan §3.3)"},
	{Repo: "inference-sim", Path: "cmd/", Match: "scheduler", Milestone: 4, Reason: "--scheduler (inference-sim #2037)"},
	{Repo: "inference-sim", Path: "cmd/", Match: "preemption-policy", Milestone: 4, Reason: "--preemption-policy (#2037)"},
	{Repo: "inference-sim", Path: "cmd/", Match: "dp-load-balancing", Milestone: 4, Reason: "--dp-load-balancing (#2037)"},
	{Repo: "inference-sim", Path: "cmd/", Match: "swa-bounded-replay", Milestone: 4, Reason: "--swa-bounded-replay (#2037)"},
	{Repo: "inference-sim", Path: "cmd/", Match: "kv-offload-config", Milestone: 4, Reason: "--kv-offload-config (#2037)"},
	{Repo: "inference-sim", Path: "", Match: "lazy-generation", Milestone: 4, Reason: "eager workload generation and --lazy-generation (#1451)"},
	{Repo: "inference-sim", Path: "", Match: "eager", Milestone: 4, Reason: "eager workload generation (#1451)"},
}
