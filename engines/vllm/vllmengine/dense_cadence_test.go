// Original to blis-vllm.

package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// vLLM v0.33.0 gives each rank of a dense (non-MoE) data-parallel deployment its own local prefill
// cadence (scheduler.py:398-403), which the data-parallel group does not model: such a deployment
// with prefill_schedule_interval above 1 is refused by name, citing issues/2656. A dense model at
// data_parallel_size 1 (the local cadence, ported) and a MoE model at data_parallel_size above 1
// (the engine-core cadence, modelled) still run, and so does a dense data-parallel deployment that
// leaves the interval at 1.
func TestDenseDataParallelLocalCadenceIsRefusedByName(t *testing.T) {
	Register()
	dense, err := os.ReadFile(filepath.Join("..", "kerneltests", "testdata", "aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp2.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	moe, err := os.ReadFile(filepath.Join("testdata", "d116-032.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	build := func(t *testing.T, base []byte, dpOld, dpNew, interval string) error {
		t.Helper()
		body := string(base)
		if !strings.Contains(body, dpOld) {
			t.Fatalf("the fixture has no %q", dpOld)
		}
		body = strings.Replace(body, dpOld, dpNew, 1)
		if interval != "" {
			body = strings.Replace(body, "      block_size: 16\n", "      block_size: 16\n      prefill_schedule_interval: "+interval+"\n", 1)
		}
		path := filepath.Join(t.TempDir(), "v.yaml")
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.34.0+0fda86a35419", Roots: map[string]string{RootScenario: path}})
		return err
	}
	for _, c := range []struct {
		name         string
		base         []byte
		dpOld, dpNew string
		interval     string
		refused      bool
	}{
		{"dense dp 2, interval 4", dense, "      dp: 1\n", "      dp: 2\n", "4", true},
		{"dense dp 2, interval 1", dense, "      dp: 1\n", "      dp: 2\n", "1", false},
		{"dense dp 1, interval 4", dense, "      dp: 1\n", "      dp: 1\n", "4", false},
		{"moe dp 4, interval 4", moe, "      dp: 4\n", "      dp: 4\n", "4", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := build(t, c.base, c.dpOld, c.dpNew, c.interval)
			if c.refused {
				if err == nil || !strings.Contains(err.Error(), "dense_dp_local_prefill_cadence") || !strings.Contains(err.Error(), "issues/2656") {
					t.Fatalf("err %v, want a refusal naming dense_dp_local_prefill_cadence and issues/2656", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("the deployment should build: %v", err)
			}
		})
	}
}
