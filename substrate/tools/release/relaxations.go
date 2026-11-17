package main

// relaxations is release plan §3.3, "Relaxing the deprecation rule until v0.18.0": the features
// LIF-7 lets stay past a compatibility-breaking release until the M3 milestone release removes
// them. Each row's removal is an M3 item: blis-vllm's compat/ and rules (inference-sim #2027),
// contracts/rules (blis-substrate #165) and contracts/kernel/enginesettings (#58). The release
// bot's deprecation check skips exactly these. Every other deprecated feature, including
// observe, calibrate, --metrics-path and --results-path, is removed in M1.
//
// This list is the plan's, copied because the plan is not a file the bot can read. When the
// plan changes, this list changes in the same pull request; TestRelaxationsMatchThePlan
// pins it, and a reviewer compares the two.
var relaxations = []Relaxation{
	{Repo: "blis-vllm", Path: "compat/", Milestone: 3, Reason: "blis-vllm's compat/ (plan §3.3)"},
	{Repo: "blis-vllm", Path: "rules/v0_29/", Milestone: 3, Reason: "rules/v0_29 (plan §3.3)"},
	{Repo: "blis-vllm", Path: "rules/v0_31/", Milestone: 3, Reason: "rules/v0_31 and the v0.31 resolution code (plan §3.3)"},
	{Repo: "blis-substrate", Path: "contracts/rules/", Milestone: 3, Reason: "contracts/rules (plan §3.3)"},
}
