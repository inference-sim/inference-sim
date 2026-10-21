package cmd

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// #1910: the scenario's Deployment.offload block is the run's KV offload configuration; a
// --kv-offload-config file may only restate it.

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
		block, err := kvOffloadBlockFromDeployment(&o)
		if err != nil {
			t.Fatalf("%+v: %v", o, err)
		}
		got, err := resolveKVOffload(block, nil, 64)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parseKVOffloadBytes([]byte("kv_offload:\n  cpu_bytes_to_use: 1099511627776\n"))
		if err != nil {
			t.Fatal(err)
		}
		want, err := resolveKVOffload(file.KVOffload, nil, 64)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%+v resolves to %+v, the equivalent file to %+v", o, got, want)
		}
		if d := offloadConfigDiff(got, want); d != "" {
			t.Errorf("offloadConfigDiff reports %q for equal configs", d)
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
			_, err := kvOffloadBlockFromDeployment(&tc.o)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want one naming %s", err, tc.want)
			}
		})
	}
	// arc is a valid vLLM policy the cache does not simulate: the shared resolver refuses it.
	block, err := kvOffloadBlockFromDeployment(&deployment.Offload{Tiers: cpu, EvictionPolicy: "arc"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolveKVOffload(block, nil, 64); err == nil || !strings.Contains(err.Error(), "eviction_policy") {
		t.Fatalf("arc: err = %v, want the eviction_policy refusal", err)
	}
	// Two configs that differ are reported field by field.
	a, _ := resolveKVOffload(&kvOffloadBlock{CPUBytesToUse: ptrInt64(1 << 30)}, nil, 64)
	b, _ := resolveKVOffload(&kvOffloadBlock{CPUBytesToUse: ptrInt64(2 << 30)}, nil, 64)
	if d := offloadConfigDiff(a, b); !strings.Contains(d, "CPUBytesToUse") || strings.Contains(d, "BlockSize") {
		t.Fatalf("diff = %q, want exactly CPUBytesToUse", d)
	}
}

func ptrInt64(v int64) *int64 { return &v }

// writeOffloadScenario writes scenario file name from dir with the offload block inserted
// before the marker (appended when the marker is ""), into a fresh directory.
func writeOffloadScenario(t *testing.T, dir, name, marker, offload string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	switch {
	case marker == "":
		body = strings.TrimRight(body, "\n") + "\n" + offload
	case strings.Contains(body, marker):
		body = strings.Replace(body, marker, offload+marker, 1)
	default:
		t.Fatalf("fixture %s has no %q", name, marker)
	}
	out := t.TempDir()
	if err := os.WriteFile(filepath.Join(out, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(body, "endpoint_picker_config: "+pdAlwaysEPPConfig) {
		withPDRouter(t, out)
	}
	return out
}

// A scenario stating its CPU tier runs byte-identically to the same scenario without the
// block given the equivalent --kv-offload-config, colocated and P/D (every engine gets its own
// tier); the exported trace headers, which record the resolved kv_offload config, match too.
// Stating both, agreeing, is accepted; disagreeing is refused naming the field. The scenario's
// 0-byte tier is refused by the schema. A scenario-offload run replays byte-identically.
func TestRunCmd_ScenarioOffloadBlock(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	const tier = "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n  eviction_policy: lru\n  tiers:\n    - tier: cpu_dram\n      bytes: 1099511627776\n"
	file := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 1099511627776\n")
	other := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 2199023255552\n")
	for _, tc := range []struct {
		name, src, scenario, marker string
		args                        []string
		replayArgs                  []string
	}{
		{"colocated", scenarios, "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml", "",
			[]string{"--num-requests", "20", "--rate", "4", "--prefix-tokens", "512", "--seed", "5", "--horizon", "600000000"},
			[]string{"--seed", "5", "--horizon", "600000000"}},
		{"P/D", writeScenarioVariant(t, "connector: NixlConnector", "connector: MultiConnector"), "variant.yaml", "pd_transfer:",
			append(append([]string(nil), pdTopology...), "--num-requests", "20", "--rate", "4", "--prefix-tokens", "512"),
			pdTopology},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withBlock := writeOffloadScenario(t, tc.src, tc.scenario, tc.marker, tier)
			// pdTopology names its own --scenarios and --scenario; later flags win, so append ours.
			run := func(dir, prefix string, extra ...string) (string, string, error) {
				args := append(append([]string{"run", "--trace-output", prefix}, tc.args...), extra...)
				return runKernelCLI(t, append(args, "--scenarios", dir, "--scenario", tc.scenario)...)
			}
			tmp := t.TempDir()
			scen, stderr, err := run(withBlock, filepath.Join(tmp, "scen"))
			if err != nil {
				t.Fatalf("scenario offload run: %v\n%s", err, lastLines(stderr, 3))
			}
			flag, stderr, err := run(tc.src, filepath.Join(tmp, "flag"), "--kv-offload-config", file)
			if err != nil {
				t.Fatalf("flag offload run: %v\n%s", err, lastLines(stderr, 3))
			}
			if scen != flag {
				t.Errorf("the scenario's offload block and the equivalent --kv-offload-config differ\n--- scenario\n%s\n--- flag\n%s", scen, flag)
			}
			hs, hf := readFile(t, filepath.Join(tmp, "scen.yaml")), readFile(t, filepath.Join(tmp, "flag.yaml"))
			if !strings.Contains(hs, "cpu_bytes_to_use: 1099511627776") || hs != hf {
				t.Errorf("trace headers differ or carry no kv_offload:\n--- scenario\n%s\n--- flag\n%s", hs, hf)
			}
			both, stderr, err := run(withBlock, filepath.Join(tmp, "both"), "--kv-offload-config", file)
			if err != nil || both != scen {
				t.Errorf("an agreeing flag beside the scenario block must change nothing: err=%v\n%s", err, lastLines(stderr, 3))
			}
			// The scenario cannot state blocks_per_chunk (no blis-schemas field), so a flag beside
			// the block supplies it, exactly as the flag alone does; it runs and replays.
			chunked := writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 1099511627776\n  blocks_per_chunk: 4\n")
			c4, stderr, err := run(withBlock, filepath.Join(tmp, "c4"), "--kv-offload-config", chunked)
			if err != nil {
				t.Fatalf("blocks_per_chunk 4 beside the scenario block: %v\n%s", err, lastLines(stderr, 3))
			}
			if h := readFile(t, filepath.Join(tmp, "c4.yaml")); !strings.Contains(h, "blocks_per_chunk: 4") {
				t.Errorf("the trace header does not record blocks_per_chunk 4:\n%s", h)
			}
			if alone, _, err := run(tc.src, filepath.Join(tmp, "c4flag"), "--kv-offload-config", chunked); err != nil || alone != c4 {
				t.Errorf("blocks_per_chunk 4 from the flag alone differs from the flag beside the scenario block: err=%v", err)
			}
			if rep, stderr, err := runKernelCLI(t, append(append([]string{"replay", "--trace-header", filepath.Join(tmp, "c4.yaml"),
				"--trace-data", filepath.Join(tmp, "c4.csv")}, tc.replayArgs...), "--scenarios", withBlock, "--scenario", tc.scenario, "--kv-offload-config", chunked)...); err != nil || rep != c4 {
				t.Errorf("the blocks_per_chunk 4 run's replay differs: err=%v\n%s", err, lastLines(stderr, 3))
			}
			if _, stderr, err := run(withBlock, filepath.Join(tmp, "bad"), "--kv-offload-config", other); err == nil ||
				!strings.Contains(stderr, "disagrees with scenario") || !strings.Contains(stderr, "CPUBytesToUse") {
				t.Errorf("a disagreeing flag must be refused naming the field: err=%v\n%s", err, lastLines(stderr, 3))
			}
			rep, stderr, err := runKernelCLI(t, append(append([]string{"replay", "--trace-header", filepath.Join(tmp, "scen.yaml"),
				"--trace-data", filepath.Join(tmp, "scen.csv")}, tc.replayArgs...), "--scenarios", withBlock, "--scenario", tc.scenario)...)
			if err != nil || rep != scen {
				t.Errorf("the scenario-offload run's replay differs: err=%v\n%s", err, lastLines(stderr, 3))
			}
			bigger := writeOffloadScenario(t, tc.src, tc.scenario, tc.marker, strings.Replace(tier, "bytes: 1099511627776", "bytes: 2199023255552", 1))
			if _, stderr, err := runKernelCLI(t, append(append([]string{"replay", "--trace-header", filepath.Join(tmp, "scen.yaml"),
				"--trace-data", filepath.Join(tmp, "scen.csv")}, tc.replayArgs...), "--scenarios", bigger, "--scenario", tc.scenario)...); err == nil ||
				!strings.Contains(stderr, "conflicts with the kv_offload config recorded in the trace header") {
				t.Errorf("replaying under a scenario whose tier differs from the trace's must be refused: err=%v\n%s", err, lastLines(stderr, 3))
			}
			zero := writeOffloadScenario(t, tc.src, tc.scenario, tc.marker, strings.Replace(tier, "bytes: 1099511627776", "bytes: 0", 1))
			if _, stderr, err := run(zero, filepath.Join(tmp, "zero")); err == nil || !strings.Contains(stderr, "must be positive") {
				t.Errorf("a 0-byte tier must be refused by the schema: err=%v\n%s", err, lastLines(stderr, 3))
			}
		})
	}
}

// A secondary tier in the scenario is vLLM's fs tier behind the CPU tier, as a
// TieringOffloadingSpec secondary_tiers [{type: fs, root_dir}] entry runs it: the device class
// prices it, vLLM's 16+16 thread defaults serve it, it runs O_DIRECT, and it holds the stated
// bytes. It resolves as the equivalent kv_offload file does.
func TestKVOffloadFromDeployment_ASecondaryTierIsTheFSTier(t *testing.T) {
	o := deployment.Offload{Spec: "TieringOffloadingSpec", Tiers: []deployment.Tier{
		{Device: "cpu_dram", Bytes: 1 << 30}, {Device: "nvme_gen4", Bytes: 1 << 40}}}
	block, err := kvOffloadBlockFromDeployment(&o)
	if err != nil {
		t.Fatal(err)
	}
	devices := map[string]kvOffloadDevice{"nvme_gen4": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80}}
	got, err := resolveKVOffload(block, devices, 64)
	if err != nil {
		t.Fatal(err)
	}
	file, err := parseKVOffloadBytes([]byte("kv_offload:\n  cpu_bytes_to_use: 1073741824\n  secondary_tiers:\n" +
		"    - {type: fs, root_dir: " + fsTierRoot + ", direct_io: true, device_class: nvme_gen4, capacity_bytes: 1099511627776}\n"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := resolveKVOffload(file.KVOffload, devices, 64)
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

// The SimpleCPUOffloadConnector is accepted: one block-granular CPU tier, every block offloaded
// (no prompt-only clamp), eager by default and lazy when the spec says so; anything else it
// cannot be is refused.
func TestKVOffloadFromDeployment_TheSimpleConnector(t *testing.T) {
	cpu := []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 30}}
	for _, tc := range []struct {
		spec string
		lazy bool
	}{{"", false}, {"eager", false}, {"lazy", true}} {
		block, err := kvOffloadBlockFromDeployment(&deployment.Offload{Connector: simpleConnector, Spec: tc.spec, Tiers: cpu})
		if err != nil {
			t.Fatalf("spec %q: %v", tc.spec, err)
		}
		got, err := resolveKVOffload(block, nil, 64)
		if err != nil {
			t.Fatalf("spec %q: %v", tc.spec, err)
		}
		if !got.Simple || got.LazyOffload != tc.lazy || got.OffloadPromptOnly || got.BlocksPerChunk != 1 {
			t.Fatalf("spec %q resolves to %+v", tc.spec, got)
		}
		back := headerToSimOffload(simToHeaderOffload(got))
		if back.Simple != got.Simple || back.LazyOffload != got.LazyOffload {
			t.Fatalf("spec %q: the trace header loses the connector mode", tc.spec)
		}
	}
	for _, o := range []deployment.Offload{
		{Connector: simpleConnector, Spec: "TieringOffloadingSpec", Tiers: cpu},
		{Connector: simpleConnector, Tiers: append(cpu, deployment.Tier{Device: "nvme_gen4", Bytes: 1 << 40})},
	} {
		if _, err := kvOffloadBlockFromDeployment(&o); err == nil {
			t.Fatalf("%+v must be refused", o)
		}
	}
}
