package cmd

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/inference-sim/sim"
)

// The scenario's offload block (#1910). A blis-schemas Deployment states its KV offload
// hierarchy as
//
//	offload: {tiers: [{tier: <catalog device class>, bytes}], connector, spec,
//	          eviction_policy, prefetch_depth}
//
// and BLIS simulates vLLM's OffloadingConnector, whose CPU tier is TieringOffloadingSpec's
// cpu_bytes_to_use (vllm/v1/kv_offload/cpu/spec.py:97-120 and tiering/spec.py at v0.31.0).
// kvOffloadBlockFromDeployment maps the block onto that model's kv_offload terms and the
// shared resolver applies vLLM's defaults to everything the schema does not state; a field the
// model does not implement is refused by name rather than dropped (R1).
//
// The block is deployment-level and the schema has no per-pool form, while vLLM's
// cpu_bytes_to_use belongs to one engine's connector config. So every engine -- the one pool
// of a colocated run, each prefill and each decode engine of a P/D run -- gets its own tier of
// the stated bytes (kernelPools.applyOffload sizes it by that pool's bytes per block).

// offloadCPUTier is the catalog device class that names the CPU (host DRAM) tier: the
// OffloadingConnector's primary tier, sized by cpu_bytes_to_use.
const offloadCPUTier = kernelCPUTier

// offloadConnectors are the offload connectors the model is: vLLM's native OffloadingConnector
// (the connector `--kv-offloading-backend native` configures, vllm/config/vllm.py:1140-1148).
// SimpleCPUOffloadConnector and LMCache place and evict KV on other terms.
var offloadConnectors = map[string]bool{"": true, "OffloadingConnector": true}

// offloadSpecs are the OffloadingConnector specs the model is. With only a CPU tier the two
// registered specs (rules v0_31 OffloadSpecs) keep the same CPU tier.
var offloadSpecs = map[string]bool{"": true, "CPUOffloadingSpec": true, "TieringOffloadingSpec": true}

// kvOffloadBlockFromDeployment maps a scenario's offload block onto the kv_offload model, or
// names what the model cannot represent. The schema has already refused a tier of no bytes.
func kvOffloadBlockFromDeployment(o *deployment.Offload) (*kvOffloadBlock, error) {
	if !offloadConnectors[o.Connector] {
		return nil, fmt.Errorf("offload.connector %q is not simulated: BLIS models vLLM's "+
			"OffloadingConnector, so state it (or leave the field out)", o.Connector)
	}
	if !offloadSpecs[o.Spec] {
		return nil, fmt.Errorf("offload.spec %q is not simulated: BLIS models the OffloadingConnector's "+
			"CPUOffloadingSpec and TieringOffloadingSpec", o.Spec)
	}
	if o.PrefetchDepth != 0 {
		return nil, fmt.Errorf("offload.prefetch_depth %d is not simulated: a tier fetch is issued "+
			"when batch formation needs it, so leave prefetch_depth out", o.PrefetchDepth)
	}
	block := &kvOffloadBlock{}
	for i, t := range o.Tiers {
		if t.Device != offloadCPUTier {
			return nil, fmt.Errorf("offload.tiers[%d] (%q) is not read from the scenario: only the "+
				"%s tier is -- a secondary tier needs the fs tier's root_dir, direct_io and thread "+
				"counts, which blis-schemas does not state, so describe it with --kv-offload-config "+
				"and drop the tier from the scenario", i, t.Device, offloadCPUTier)
		}
		b := t.Bytes
		block.CPUBytesToUse = &b
	}
	if block.CPUBytesToUse == nil {
		return nil, fmt.Errorf("offload states no %s tier: the OffloadingConnector's primary tier is "+
			"host DRAM (cpu_bytes_to_use, required by vLLM)", offloadCPUTier)
	}
	if o.EvictionPolicy != "" {
		ep := o.EvictionPolicy
		block.EvictionPolicy = &ep
	}
	return block, nil
}

// offloadConfigDiff names the fields in which two resolved offload configs differ, "" when
// they agree. PerBlockBytes is not compared: it is derived from the kernel after resolution.
func offloadConfigDiff(a, b sim.KVOffloadConfig) string {
	a.PerBlockBytes, b.PerBlockBytes = 0, 0
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	var diffs []string
	for i := 0; i < va.NumField(); i++ {
		if !reflect.DeepEqual(va.Field(i).Interface(), vb.Field(i).Interface()) {
			name := va.Type().Field(i).Name
			diffs = append(diffs, fmt.Sprintf("%s: scenario %v, flag %v", name, va.Field(i).Interface(), vb.Field(i).Interface()))
		}
	}
	return strings.Join(diffs, "; ")
}
