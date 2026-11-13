// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/config/parallel.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// ParallelConfig.use_ubatching and num_ubatches (:636-646) and set_dcp_defaults (:607-620). Also
// vllm/v1/worker/ubatch_utils.py (check_ubatch_thresholds, :140-148) and
// vllm/model_executor/models/config.py (GlmMoeDsaForCausalLM, :66-70).

package lower

import "fmt"

// DBO is the dual-batch-overlap settings of a pool.
type DBO struct {
	// Enable is enable_dbo, and UBatchSize ubatch_size.
	Enable     bool
	UBatchSize int
	// DecodeThreshold and PrefillThreshold are dbo_decode_token_threshold and
	// dbo_prefill_token_threshold, with vLLM's defaults (32 and 512) already applied by the caller.
	DecodeThreshold, PrefillThreshold int
}

// Ubatches is how many micro-batches a step is split into when it splits: num_ubatches when
// ubatching is on (enable_dbo, or ubatch_size above 1), else 1.
func (d DBO) Ubatches() int {
	if !d.Enable && d.UBatchSize <= 1 {
		return 1
	}
	if d.Enable {
		return 2
	}
	return d.UBatchSize
}

// Splits is check_ubatch_thresholds: a step splits when ubatching is on and its token count
// reaches the decode threshold for a uniform-decode batch, else the prefill threshold.
func (d DBO) Splits(tokens int, uniformDecode bool) bool {
	if !d.Enable && d.UBatchSize <= 1 {
		return false
	}
	if uniformDecode {
		return tokens >= d.DecodeThreshold
	}
	return tokens >= d.PrefillThreshold
}

// DCP is the decode-context-parallel communication settings a pool resolves to.
type DCP struct {
	// Backend is "ag_rs" or "a2a"; QReplicate is dcp_q_replicate.
	Backend    string
	QReplicate bool
}

// DCPModelPreference is what a model's verify_and_update_config hook passes to set_dcp_defaults.
// The stock preference (ag_rs, no query replication) applies when the model sets none. In v0.32.0
// only GlmMoeDsaForCausalLM sets one: a2a with query replication (models/config.py:66-70).
func DCPModelPreference(architecture string) DCP {
	if architecture == "GlmMoeDsaForCausalLM" {
		return DCP{Backend: "a2a", QReplicate: true}
	}
	return DCP{Backend: "ag_rs"}
}

// DCPResolve is set_dcp_defaults: a stated value is kept and an unstated one takes the model's
// preference. A stated backend outside the two names is refused (the Literal of parallel.py:40).
func DCPResolve(statedBackend string, statedQReplicate *bool, model DCP) (DCP, error) {
	out := model
	if statedBackend != "" {
		if statedBackend != "ag_rs" && statedBackend != "a2a" {
			return DCP{}, fmt.Errorf("lower: dcp_comm_backend %q is not ag_rs or a2a (vllm/config/parallel.py:40)", statedBackend)
		}
		out.Backend = statedBackend
	}
	if statedQReplicate != nil {
		out.QReplicate = *statedQReplicate
	}
	return out, nil
}
