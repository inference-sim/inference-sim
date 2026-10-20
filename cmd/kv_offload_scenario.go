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

// fsTierRoot is the root_dir a scenario's secondary tier is given. blis-schemas states a tier as
// a catalog device and its bytes, not a directory; the directory is only the tier's identity in
// vLLM (vllm/v1/kv_offload/file_mapper.py:135-144 at v0.31.0), and BLIS shares a node's tier
// among that node's engines by position (sim.OffloadTierSharing), so one fixed name stands for
// the device's mount.
const fsTierRoot = "/mnt/kv-offload"

// objectStoreDevices are catalog devices that are not a filesystem: vLLM reaches them through
// its obj tier (tiering/obj), which BLIS does not simulate.
var objectStoreDevices = map[string]bool{"s3": true}

// offloadConnectors are the offload connectors the model is: vLLM's native OffloadingConnector
// (the connector `--kv-offloading-backend native` configures, vllm/config/vllm.py:1140-1148).
// SimpleCPUOffloadConnector and LMCache place and evict KV on other terms.
var offloadConnectors = map[string]bool{"": true, "OffloadingConnector": true, simpleConnector: true}

// simpleConnector is vLLM's SimpleCPUOffloadConnector (vllm/v1/simple_kv_offload at v0.31.0):
// one block-granular CPU tier whose stores are asynchronous copies. Its cpu_bytes_to_use is
// per engine and divided among the engine's workers (simple_cpu_offload_connector.py:85-137),
// as the OffloadingConnector's is; the scenario tier's bytes are the engine's.
const simpleConnector = "SimpleCPUOffloadConnector"

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
	if o.Connector == simpleConnector {
		return simpleOffloadBlock(o)
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
		if t.Device == offloadCPUTier {
			b := t.Bytes
			block.CPUBytesToUse = &b
			continue
		}
		if objectStoreDevices[t.Device] {
			return nil, fmt.Errorf("offload.tiers[%d] (%q) is an object store, which vLLM reaches through "+
				"its obj secondary tier; BLIS simulates the fs tier only", i, t.Device)
		}
		if o.Spec != "TieringOffloadingSpec" {
			return nil, fmt.Errorf("offload.tiers[%d] (%q) is a secondary tier, which only the "+
				"TieringOffloadingSpec runs (vllm/v1/kv_offload/tiering/spec.py); state "+
				"offload.spec: TieringOffloadingSpec", i, t.Device)
		}
		block.SecondaryTiers = append(block.SecondaryTiers, fsTierBlock(t))
	}
	if len(block.SecondaryTiers) > 0 && (len(o.Tiers) == 0 || o.Tiers[0].Device != offloadCPUTier) {
		return nil, fmt.Errorf("offload.tiers[0] must be the %s tier: every secondary tier sits behind "+
			"the CPU primary tier (tiering/manager.py:1-20)", offloadCPUTier)
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

// fsTierBlock is the fs secondary tier a scenario tier states: vLLM's fs tier on the device
// (secondary_tiers [{type: fs, root_dir}], tiering/fs/manager.py:109-120 at v0.31.0) with its
// default 16 read and 16 write threads, holding up to the stated bytes. It runs O_DIRECT, which
// vLLM probes and uses where the filesystem accepts it (fs/manager.py:191-201) -- a local NVMe
// ext4 or xfs mount does -- and which is the regime the kernel prices.
func fsTierBlock(t deployment.Tier) kvOffloadTierBlock {
	fs, root, direct, class := "fs", fsTierRoot, true, t.Device
	capacity := t.Bytes
	return kvOffloadTierBlock{Type: &fs, RootDir: &root, DirectIO: &direct, DeviceClass: &class,
		CapacityBytes: &capacity}
}

// simpleOffloadBlock maps a SimpleCPUOffloadConnector block: one cpu_dram tier, block-granular,
// every block offloaded (its eager store has no prompt-only clamp, manager.py:641-951 at
// v0.31.0). The connector takes no spec; its lazy_offload mode is OffloadSpec "lazy" here,
// since blis-schemas has no field for it (eager, vLLM's default, when unstated).
func simpleOffloadBlock(o *deployment.Offload) (*kvOffloadBlock, error) {
	if o.Spec != "" && o.Spec != "lazy" && o.Spec != "eager" {
		return nil, fmt.Errorf("offload.spec %q: the SimpleCPUOffloadConnector has no offloading spec; "+
			"state \"lazy\" for lazy_offload, \"eager\" or nothing for its default", o.Spec)
	}
	if len(o.Tiers) != 1 || o.Tiers[0].Device != offloadCPUTier {
		return nil, fmt.Errorf("offload: the SimpleCPUOffloadConnector has one %s tier (its disk backend "+
			"replaces the CPU tier and is not simulated)", offloadCPUTier)
	}
	if o.PrefetchDepth != 0 {
		return nil, fmt.Errorf("offload.prefetch_depth %d is not simulated", o.PrefetchDepth)
	}
	b := o.Tiers[0].Bytes
	promptOnly, one, simple, lazy := false, int64(1), true, o.Spec == "lazy"
	block := &kvOffloadBlock{CPUBytesToUse: &b, OffloadPromptOnly: &promptOnly, BlocksPerChunk: &one,
		Simple: &simple, LazyOffload: &lazy}
	if o.EvictionPolicy != "" && o.EvictionPolicy != "lru" {
		return nil, fmt.Errorf("offload.eviction_policy %q: the SimpleCPUOffloadConnector's CPU pool is LRU", o.EvictionPolicy)
	}
	return block, nil
}
