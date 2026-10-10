package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/spf13/cobra"
)

// sizedKnobs is every engine knob adoptKernelDeployment sizes the run from.
type sizedKnobs struct {
	kvBlocks, blockSize, maxNumSeqs, maxBatched, maxModelLen int64
	specTokens, dp                                           int
	noPrefixCaching                                          bool
}

func currentSizedKnobs() sizedKnobs {
	return sizedKnobs{totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen,
		numSpeculativeTokens, dataParallelism, noEnablePrefixCaching}
}

// Replay is sized from the kernel exactly as run is: parsing replay's flag set (the shared sim
// config flags plus the trace inputs) and adopting the deployment yields, for every engine knob,
// the kernel's Settings answer -- and the same answer run's flag set yields. Checked on a dense
// and a dp=2 MoE scenario, where the per-rank budget and the aggregate differ.
func TestAdoptKernelDeployment_SizesTheReplayFromTheKernel(t *testing.T) {
	scenarios, catalog, registry := kernelRepos(t)
	setupKernelTestFixtures(t)
	saved := currentSizedKnobs()
	savedModel, savedGPU, savedTP, savedEP := model, gpu, tensorParallelism, enableExpertParallel
	t.Cleanup(func() {
		totalKVBlocks, blockSizeTokens, maxNumSeqs = saved.kvBlocks, saved.blockSize, saved.maxNumSeqs
		maxNumBatchedTokens, maxModelLen, numSpeculativeTokens = saved.maxBatched, saved.maxModelLen, saved.specTokens
		dataParallelism, noEnablePrefixCaching = saved.dp, saved.noPrefixCaching
		model, gpu, tensorParallelism, enableExpertParallel = savedModel, savedGPU, savedTP, savedEP
	})
	dir := t.TempDir()
	header, data := filepath.Join(dir, "t.yaml"), filepath.Join(dir, "t.csv")
	for _, p := range []string{header, data} {
		if err := os.WriteFile(p, []byte("x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	adopt := func(t *testing.T, replay bool, scenario string) sizedKnobs {
		t.Helper()
		cmd := &cobra.Command{}
		registerSimConfigFlags(cmd)
		args := []string{"--scenario", scenario, "--scenarios", scenarios, "--registry", registry, "--catalog", catalog}
		if replay {
			cmd.Flags().String("trace-header", "", "")
			cmd.Flags().String("trace-data", "", "")
			args = append(args, "--trace-header", header, "--trace-data", data)
		}
		if err := cmd.ParseFlags(args); err != nil {
			t.Fatal(err)
		}
		adoptKernelDeployment(cmd)
		return currentSizedKnobs()
	}
	for _, scenario := range []string{
		"llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml",
		"minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml",
	} {
		t.Run(scenario, func(t *testing.T) {
			m, err := kernelmodel.Open(scenario, kernelmodel.Repos{Scenarios: scenarios, Catalog: catalog, Registry: registry})
			if err != nil {
				t.Fatal(err)
			}
			st, err := m.Settings()
			if err != nil {
				t.Fatal(err)
			}
			want := sizedKnobs{st.KVBlocks, int64(st.BlockSize), int64(st.MaxNumSeqs), int64(st.MaxNumBatchedTokens),
				int64(st.MaxModelLen), st.SpeculativeTokens, st.DataParallel, st.PrefixCachingDisabled}
			// Poison the knobs first so a replay adoption that sized nothing cannot pass on the
			// values a previous adoption left behind.
			totalKVBlocks, maxNumSeqs, maxNumBatchedTokens, maxModelLen = -1, -1, -1, -1
			replayed := adopt(t, true, scenario)
			if replayed != want {
				t.Errorf("replay sizes the run at %+v; the kernel answers %+v", replayed, want)
			}
			if run := adopt(t, false, scenario); run != replayed {
				t.Errorf("run sizes at %+v, replay at %+v", run, replayed)
			}
			if st.DataParallel > 1 && st.AggregateKVBlocks <= st.KVBlocks {
				t.Errorf("dp %d: aggregate %d is not above per-rank %d, so this case does not distinguish them",
					st.DataParallel, st.AggregateKVBlocks, st.KVBlocks)
			}
		})
	}
}

// writeLoRAPressureFixtures writes a colocated variant whose KV pool is small (a low
// gpu_memory_utilization), a LoRA config whose adapters reserve HBM beside the weights, and a
// workload whose requests use those adapters with long prompts: together they overflow the
// KV pool, so the run preempts.
func writeLoRAPressureFixtures(t *testing.T) (scenarioDir, loraPath, specPath string) {
	t.Helper()
	scenarioDir = writeColocatedVariant(t, "gpu_memory_utilization: 0.9", "gpu_memory_utilization: 0.25")
	dir := t.TempDir()
	loraPath = filepath.Join(dir, "lora.yaml")
	if err := os.WriteFile(loraPath, []byte("lora:\n  adapter_capacity: 2\n  adapters:\n"+
		"    - id: a0\n      rank: 64\n    - id: a1\n      rank: 64\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	b.WriteString("version: \"2\"\ncategory: language\naggregate_rate: 40.0\nnum_requests: 120\nclients:\n")
	for i, adapter := range []string{"a0", "a1"} {
		fmt.Fprintf(&b, "  - id: c%d\n    tenant_id: t\n    slo_class: batch\n    adapter: %s\n    rate_fraction: 1.0\n"+
			"    arrival:\n      process: poisson\n"+
			"    input_distribution:\n      type: constant\n      params: { value: 6000 }\n"+
			"    output_distribution:\n      type: constant\n      params: { value: 600 }\n", i, adapter)
	}
	specPath = filepath.Join(dir, "spec.yaml")
	if err := os.WriteFile(specPath, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return scenarioDir, loraPath, specPath
}

// A run with LoRA adapters under KV pressure -- the reservation shrinking the pool, requests
// preempted -- replays byte-identically from its exported trace with the same flags and horizon
// (INV-13): replay sets aside the same reservation and preempts the same requests. Non-vacuity:
// the run preempts, so a replay sized without the reservation would diverge.
func TestReplayCmd_KernelBackend_ReproducesALoRARunUnderKVPressure(t *testing.T) {
	dir, lora, spec := writeLoRAPressureFixtures(t)
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--scenarios", dir, "--scenario", "variant.yaml", "--lora-config", lora,
		"--seed", "5", "--horizon", "6000000000"}
	runOut, stderr, err := runKernelCLI(t, append([]string{"run", "--workload-spec", spec, "--trace-output", prefix}, common...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	if got := clusterMetricInt(t, runOut, "preemption_count"); got <= 0 {
		t.Fatalf("the run preempted %d times; the fixture no longer puts the KV pool under pressure", got)
	}
	if got := clusterMetricInt(t, runOut, "completed_requests"); got <= 0 {
		t.Fatalf("the run completed %d requests", got)
	}
	repOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
		"--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	if repOut != runOut {
		t.Errorf("the LoRA run's replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, repOut)
	}
}
