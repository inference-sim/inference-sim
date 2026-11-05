package resolve

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// The decode-context-parallel combine backends this kernel prices. They are the engine's
// DCPCommBackend literal at v0.31.0 (vllm/config/parallel.py:40), and each launches a
// different set of collectives per decode layer; see dcpDecodeCollectives.
const (
	// DCPAllGatherReduceScatter is the engine default: a log-sum-exp all-gather, then an
	// output reduce-scatter (vllm/v1/attention/ops/dcp.py:438-500).
	DCPAllGatherReduceScatter = "ag_rs"
	// DCPAllToAll packs the partial output and its log-sum-exp into one buffer and
	// exchanges it with a single all-to-all (dcp.py:939-1010).
	DCPAllToAll = "a2a"
)

// DecodeContext is how a decode-context-parallel group will run, after resolution.
type DecodeContext struct {
	// CommBackend is the combine that runs: DCPAllGatherReduceScatter or DCPAllToAll.
	CommBackend string
	// Interleave is how many consecutive tokens one rank holds before the next takes
	// over (cp_kv_cache_interleave_size), as it will run.
	Interleave int
}

// ResolveDecodeContext settles two of the three decode-context-parallel engine knobs that
// blis-schemas v0.2.2 made expressible -- the combine backend and the interleave -- the way
// vLLM v0.31.0 settles them. The third, dcp_q_replicate, is refused where it would take
// effect, by the kernel's resolveContextParallel.
//
// accepted is the set of backend names the scenario's engine release accepts, from its
// rules pack; nil means the caller has no pack to consult, and only the names this kernel
// can price are checked. blockSize is the KV block size the kernel resolved, and blockFinal
// whether it is the size the engine will run. A stated interleave that does not fit the block
// is refused either way: against a final block because the engine refuses it, and against
// one the kernel assumed because it cannot tell whether the engine would -- the platform
// usually keeps the default 16 and may raise it -- so the deployment must state the block.
//
// What it does NOT settle is the model's own preference. An unstated backend is the stock
// default here, ag_rs, but a model's configuration hook may choose otherwise before the
// stock default applies (ParallelConfig.set_dcp_defaults, parallel.py:581-594, called from
// vllm/config/vllm.py:1434-1440) -- GlmMoeDsaForCausalLM selects a2a with a replicated query
// projection (vllm/model_executor/models/config.py:43-50). The kernel holds no model
// identity to dispatch on, so a deployment of such a model must state the backend it runs;
// the caller records the default as an assumption so the gap is visible.
func ResolveDecodeContext(pool deployment.Pool, e enginesettings.Settings, dep *deployment.Deployment, blockSize int,
	blockFinal bool, accepted map[string]bool) (DecodeContext, []Override, error) {
	dcp, pcp := pool.Parallel.DCP, pool.Parallel.PCP
	out := DecodeContext{CommBackend: DCPAllGatherReduceScatter, Interleave: 1}

	// The backend is a closed literal the engine validates whatever the DCP width, so an
	// unknown name is refused even where it would have no effect.
	if b := e.DCPCommBackend; b != "" {
		if accepted != nil && !accepted[b] {
			return DecodeContext{}, nil, fmt.Errorf(
				"dcp_comm_backend %q is not one this engine release accepts (%s)",
				b, strings.Join(sortedNames(accepted), ", "))
		}
		if b != DCPAllGatherReduceScatter && b != DCPAllToAll {
			return DecodeContext{}, nil, fmt.Errorf(
				"dcp_comm_backend %q is not one this kernel prices; add its collectives "+
					"to the DCP combine rather than letting it fall through", b)
		}
		out.CommBackend = b
	}
	if dcp <= 1 {
		// Nothing is sharded, so neither the combine nor the stripe exists.
		return out, nil, nil
	}
	// Where PCP with DCP runs at all (the DSA layers that opt in; New refuses the rest), the
	// combine is an all-gather and an all-reduce, and the engine refuses the all-to-all
	// there outright: "MRV2 PCP + DCP requires dcp_comm_backend='ag_rs'"
	// (vllm/v1/worker/gpu/pcp_manager.py:188-194).
	if pcp > 1 && out.CommBackend == DCPAllToAll {
		return DecodeContext{}, nil, fmt.Errorf(
			"dcp_comm_backend a2a with pcp %d: the engine requires ag_rs when prefill- "+
				"and decode-context parallelism are combined", pcp)
	}

	var overrides []Override
	switch stated := e.CPKVCacheInterleaveSize; {
	case stated > 0:
		// Stated: v0.31.0 honours it, NIXL or not. Only an unstated size is auto-resolved
		// (_allow_auto_resolve_cp_interleave_size is set exactly when the flag is omitted,
		// vllm/engine/arg_utils.py:2509-2516).
		out.Interleave = stated
		// And it must fit the block: no larger, and dividing it, unless NIXL P/D is
		// configured, where each worker pins it instead (VllmConfig.validate_block_size,
		// vllm/config/vllm.py:3382-3404; ParallelConfig, vllm/config/parallel.py:402-403).
		if fits := stated <= blockSize && blockSize%stated == 0; !fits {
			names := connectorNames(dep)
			switch {
			case names["MultiConnector"]:
				return DecodeContext{}, nil, fmt.Errorf(
					"cp_kv_cache_interleave_size %d does not fit block_size %d, which the "+
						"engine refuses unless NixlConnector is configured, and a "+
						"MultiConnector's children cannot be named in a deployment",
					stated, blockSize)
			case names["NixlConnector"]:
			case blockFinal:
				return DecodeContext{}, nil, fmt.Errorf(
					"cp_kv_cache_interleave_size %d must be no larger than block_size %d "+
						"and divide it; the engine refuses this layout at startup",
					stated, blockSize)
			default:
				return DecodeContext{}, nil, fmt.Errorf(
					"cp_kv_cache_interleave_size %d does not fit block_size %d; the engine "+
						"refuses it if that is the block it runs, which the kernel cannot "+
						"confirm -- an unstated block is assumed, and a hybrid model's may be "+
						"re-aligned by the platform -- so the layout is refused rather than "+
						"priced", stated, blockSize)
			}
		}
	default:
		pinned, err := nixlPinsInterleave(dep)
		if err != nil {
			return DecodeContext{}, nil, err
		}
		if pinned && blockSize > 1 {
			// With NIXL configured, the engine sets an unstated size to the LOCAL block
			// size -- deliberately not the dcp-scaled one -- for block-level alignment
			// across the transfer (vllm/config/vllm.py:3333-3378). Recorded, because a
			// reader of the deployment sees no interleave at all.
			out.Interleave = blockSize
			overrides = append(overrides, Override{
				Field:     "cp_kv_cache_interleave_size",
				Requested: "unstated (engine default 1)",
				Resolved:  fmt.Sprint(blockSize),
				Reason: "with NixlConnector configured and the size unstated, the " +
					"engine aligns the decode-context stripe to the KV block so " +
					"transferred blocks stay whole",
			})
		}
	}
	return out, overrides, nil
}

// nixlPinsInterleave reports whether the deployment configures the NIXL connector, which
// is the only one that auto-resolves the interleave at v0.31.0 (vllm/config/vllm.py:3356-3360
// returns early unless kv_transfer_config.has_connector("NixlConnector")).
//
// A deployment names a connector for prefill-to-decode transfer and one for offload, and
// the engine's has_connector also looks inside a MultiConnector's children
// (vllm/config/kv_transfer.py:156-163). Those children are not expressible in a deployment,
// so a MultiConnector leaves the answer unknown, and that is an error rather than a guess:
// guessing 1 or the block size misstates a rank's share of every sequence.
func nixlPinsInterleave(dep *deployment.Deployment) (bool, error) {
	names := connectorNames(dep)
	if names["MultiConnector"] {
		return false, fmt.Errorf(
			"cp_kv_cache_interleave_size is unstated under a MultiConnector, whose " +
				"children a deployment cannot name; the engine pins the size to the " +
				"block size only if one of them is NixlConnector, so state the size")
	}
	return names["NixlConnector"], nil
}

// connectorNames is the set of KV connectors a deployment configures: one for
// prefill-to-decode transfer and one for offload.
func connectorNames(dep *deployment.Deployment) map[string]bool {
	names := map[string]bool{}
	if dep == nil {
		return names
	}
	if dep.PDTransfer != nil {
		names[dep.PDTransfer.Connector] = true
	}
	if dep.Offload != nil {
		names[dep.Offload.Connector] = true
	}
	return names
}

func sortedNames(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
