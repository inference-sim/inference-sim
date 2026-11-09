package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
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
			m, err := kernelmodel.Open(scenario, reposOf(scenarios, catalog, registry))
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
