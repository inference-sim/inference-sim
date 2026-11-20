// Original to blis-sglang.

package lower

import "github.com/inference-sim/blis-sglang/config"

// issueCP is the issue that models context parallelism. It stays open while this refusal cites it
// (X28).
const issueCP = "inference-sim/blis-sglang#26"

// contextParallel refuses, by its SGLang name, a context-parallel setting a resolved ServerArgs
// holds away from its default: attention context parallelism (attn_cp_size, enable_prefill_cp,
// cp_strategy) and decode context parallelism (dcp_size), declared at
// python/sglang/srt/arg_groups/fields/parallel.py:52, :104, :153 and :157@297e54fd. No target
// scenario states either, so neither the KV split nor the attention gathers they imply are
// lowered. dcp_comm_backend is not read here: resolution always sets it (to ag_rs when dcp_size is
// 1), and config.Resolve refuses a user's own value of it, citing the same issue.
func contextParallel(a *config.ServerArgs) error {
	switch {
	case a.AttnCpSize != 1:
		return &NotLowered{Name: "attn_cp_size", Issue: issueCP}
	case a.DcpSize != 1:
		return &NotLowered{Name: "dcp_size", Issue: issueCP}
	case a.EnablePrefillCp:
		return &NotLowered{Name: "enable_prefill_cp", Issue: issueCP}
	case a.CpStrategy != nil:
		return &NotLowered{Name: "cp_strategy", Issue: issueCP}
	}
	return nil
}
