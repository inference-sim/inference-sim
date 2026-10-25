// Moved from inference-sim/cmd/kv_offload_scenario_test.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.

package build

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// The mapping: a cpu_dram tier of N bytes, with the OffloadingConnector, either of its specs
// and LRU, resolves to exactly the config a kv_offload file of cpu_bytes_to_use N does --
// vLLM's defaults applied to everything else alike.
func TestKVOffloadFromDeployment_ResolvesAsTheEquivalentFile(t *testing.T) {
	const n = int64(1) << 40
	for _, o := range []deployment.Offload{
		{Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: n}}},
		{Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: n}}, Connector: "OffloadingConnector",
			Spec: "TieringOffloadingSpec", EvictionPolicy: "lru"},
		{Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: n}}, Spec: "CPUOffloadingSpec"},
	} {
		block, err := KVOffloadBlockFromDeployment(&o)
		if err != nil {
			t.Fatalf("%+v: %v", o, err)
		}
		got, err := ResolveKVOffload(block, nil, 64)
		if err != nil {
			t.Fatal(err)
		}
		file, err := ParseKVOffloadBytes([]byte("kv_offload:\n  cpu_bytes_to_use: 1099511627776\n"))
		if err != nil {
			t.Fatal(err)
		}
		want, err := ResolveKVOffload(file.KVOffload, nil, 64)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%+v resolves to %+v, the equivalent file to %+v", o, got, want)
		}
		if d := OffloadConfigDiff(got, want); d != "" {
			t.Errorf("OffloadConfigDiff reports %q for equal configs", d)
		}
	}
}

// Every field the model does not implement is refused by name.
func TestKVOffloadFromDeployment_RefusesWhatTheModelDoesNotImplement(t *testing.T) {
	cpu := []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 30}}
	for _, tc := range []struct {
		name string
		o    deployment.Offload
		want string
	}{
		{"LMCache", deployment.Offload{Tiers: cpu, Connector: "LMCacheConnectorV1"}, "offload.connector"},
		{"an unknown spec", deployment.Offload{Tiers: cpu, Spec: "MySpec"}, "offload.spec"},
		{"a prefetch depth", deployment.Offload{Tiers: cpu, PrefetchDepth: 2}, "offload.prefetch_depth"},
		{"a secondary tier without the tiering spec", deployment.Offload{Tiers: append(cpu, deployment.Tier{Device: "nvme_gen4", Bytes: 1 << 40})}, `offload.tiers[1] ("nvme_gen4")`},
		{"an object store", deployment.Offload{Spec: "TieringOffloadingSpec", Tiers: append(cpu, deployment.Tier{Device: "s3", Bytes: 1 << 40})}, "obj secondary tier"},
		{"no CPU tier", deployment.Offload{Spec: "TieringOffloadingSpec", Tiers: []deployment.Tier{{Device: "nvme_gen4", Bytes: 1 << 40}}}, "offload.tiers[0]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := KVOffloadBlockFromDeployment(&tc.o)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want one naming %s", err, tc.want)
			}
		})
	}
	// arc is a valid vLLM policy the cache does not simulate: the shared resolver refuses it.
	block, err := KVOffloadBlockFromDeployment(&deployment.Offload{Tiers: cpu, EvictionPolicy: "arc"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ResolveKVOffload(block, nil, 64); err == nil || !strings.Contains(err.Error(), "eviction_policy") {
		t.Fatalf("arc: err = %v, want the eviction_policy refusal", err)
	}
	// Two configs that differ are reported field by field.
	a, _ := ResolveKVOffload(&KVOffloadBlock{CPUBytesToUse: ptrInt64(1 << 30)}, nil, 64)
	b, _ := ResolveKVOffload(&KVOffloadBlock{CPUBytesToUse: ptrInt64(2 << 30)}, nil, 64)
	if d := OffloadConfigDiff(a, b); !strings.Contains(d, "CPUBytesToUse") || strings.Contains(d, "BlockSize") {
		t.Fatalf("diff = %q, want exactly CPUBytesToUse", d)
	}
}

func ptrInt64(v int64) *int64 { return &v }

// A secondary tier in the scenario is vLLM's fs tier behind the CPU tier, as a
// TieringOffloadingSpec secondary_tiers [{type: fs, root_dir}] entry runs it: the device class
// prices it, vLLM's 16+16 thread defaults serve it, it runs O_DIRECT, and it holds the stated
// bytes. It resolves as the equivalent kv_offload file does.
func TestKVOffloadFromDeployment_ASecondaryTierIsTheFSTier(t *testing.T) {
	o := deployment.Offload{Spec: "TieringOffloadingSpec", Tiers: []deployment.Tier{
		{Device: "cpu_dram", Bytes: 1 << 30}, {Device: "nvme_gen4", Bytes: 1 << 40}}}
	block, err := KVOffloadBlockFromDeployment(&o)
	if err != nil {
		t.Fatal(err)
	}
	devices := map[string]KVOffloadDevice{"nvme_gen4": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80}}
	got, err := ResolveKVOffload(block, devices, 64)
	if err != nil {
		t.Fatal(err)
	}
	file, err := ParseKVOffloadBytes([]byte("kv_offload:\n  cpu_bytes_to_use: 1073741824\n  secondary_tiers:\n" +
		"    - {type: fs, root_dir: " + fsTierRoot + ", direct_io: true, device_class: nvme_gen4, capacity_bytes: 1099511627776}\n"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := ResolveKVOffload(file.KVOffload, devices, 64)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("scenario tier resolves to %+v, the equivalent file to %+v", got, want)
	}
	if tier := got.Tiers[0]; tier.NReadThreads != 16 || tier.NWriteThreads != 16 || !tier.DirectIO || tier.CapacityBytes != 1<<40 {
		t.Fatalf("fs tier %+v: want 16+16 threads, O_DIRECT, 1 TiB", tier)
	}
}
