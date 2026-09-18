package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/latency"
)

// blockTypeOnlyConfigJSON is a Nemotron-3-Ultra-550B-shaped config (scaled down to keep
// the test fast): the layer count exists ONLY as the length of layers_block_type — 8
// entries — with no num_hidden_layers scalar anywhere. Before #1729 this aborted the run
// with "NumLayers must be > 0, got 0" before a single request was simulated.
const blockTypeOnlyConfigJSON = `{
  "architectures": ["NemotronForCausalLM"],
  "layers_block_type": [
    "attention", "mamba", "mamba", "mamba",
    "attention", "mamba", "mamba", "mamba"
  ],
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "vocab_size": 32000,
  "hidden_act": "silu",
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`

// writeBlockTypeOnlyFixture writes the block-type-array-only config.json as the
// "nemotron-block-type-only" entry of a fresh test catalog and returns the catalog ROOT
// (what --catalog takes since #1731; the entry directory is derived from the model's
// short name).
func writeBlockTypeOnlyFixture(t *testing.T) string {
	t.Helper()
	catalogDir, err := writeTestCatalog(t.TempDir(), blockTypeOnlyConfigJSON, "nemotron-block-type-only")
	if err != nil {
		t.Fatalf("write test catalog: %v", err)
	}
	return catalogDir
}

// TestRunCmd_LayersBlockTypeOnly_Runs is BC-4 / AC-1 + AC-3 at the system level: a
// `blis run` against a model whose config expresses its layer count only as
// layers_block_type REACHES COMPLETION — it emits cluster metrics and conserves requests
// (INV-1) — on BOTH latency backends, instead of aborting at startup with
// "NumLayers must be > 0, got 0" (#1729).
func TestRunCmd_LayersBlockTypeOnly_Runs(t *testing.T) {
	if backend := os.Getenv("BLIS_BLOCKTYPE_BACKEND"); backend != "" {
		rootCmd.SetArgs([]string{
			"run",
			"--model", "nvidia/nemotron-block-type-only",
			"--catalog", os.Getenv("BLIS_BLOCKTYPE_CONFIG_DIR"),
			"--hardware", "H100",
			"--hardware-config", "../hardware_config.json",
			"--latency-model", backend,
			"--tp", "1",
			"--total-kv-blocks", "20000",
			"--rate", "10",
			"--num-requests", "20",
			"--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
		})
		// A non-fatal Execute error must surface as a distinct non-zero exit so the parent
		// does not mistake a cobra failure for the startup fatal it is testing against.
		if err := rootCmd.Execute(); err != nil {
			fmt.Fprintf(os.Stderr, "Execute failed: %v\n", err)
			os.Exit(2)
		}
		os.Exit(0)
	}

	for _, backend := range []string{"trained-physics", "roofline"} {
		t.Run(backend, func(t *testing.T) {
			cfgDir := writeBlockTypeOnlyFixture(t)
			cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_LayersBlockTypeOnly_Runs$")
			cmd.Env = append(os.Environ(),
				"BLIS_BLOCKTYPE_BACKEND="+backend,
				"BLIS_BLOCKTYPE_CONFIG_DIR="+cfgDir,
			)
			var stdout, stderr bytes.Buffer
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr
			if err := cmd.Run(); err != nil {
				t.Fatalf("AC-1: a run whose only layer-count evidence is layers_block_type must complete "+
					"on the %s backend, got err=%v\nstderr:\n%s", backend, err, stderr.String())
			}
			// AC-3: the per-backend NumLayers validation must not have fired.
			if strings.Contains(stderr.String(), "NumLayers must be > 0") {
				t.Errorf("AC-3: the %s backend still rejected the derived layer count:\n%s", backend, stderr.String())
			}
			// R1: the derivation is visible at the DEFAULT log level, naming the derived
			// count — an operator must never have to guess where the layer count came from.
			if !strings.Contains(stderr.String(), "derived NumLayers=8") {
				t.Errorf("expected the fallback diagnostic naming the derived count at the default log level "+
					"(R1: never silent); stderr:\n%s", stderr.String())
			}
			out := stdout.String()
			// 20 is this fixture's --num-requests; clusterConservationHolds compares
			// injected_requests against it, since stdout defines injected_requests as the
			// five-term sum and so cannot check that sum against itself (#1720, #1746).
			clusterConservationHolds(t, out, 20) // INV-1: the run really simulated the workload
			assertCompletedRequests(t, out)
		})
	}
}

// assertCompletedRequests fails unless the cluster aggregate reports at least one
// completed request — the non-vacuity gate for "the run reached completion". A startup
// abort emits no metrics at all; a silently-broken model would emit zeros.
func assertCompletedRequests(t *testing.T, stdout string) {
	t.Helper()
	for _, raw := range extractJSONObjects(stdout) {
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			continue
		}
		if obj["instance_id"] != "cluster" {
			continue
		}
		completed, ok := obj["completed_requests"].(float64)
		if !ok {
			t.Fatalf("cluster metrics missing numeric completed_requests")
		}
		if completed <= 0 {
			t.Errorf("AC-1: expected completed_requests > 0, got %v", completed)
		}
		return
	}
	t.Fatalf("no cluster aggregate metrics object found in stdout:\n%s", stdout)
}

// TestRunCmd_NoLayerCountEvidence_FailsLoudly is BC-3 / AC-4 at the system level: a
// config with NEITHER num_hidden_layers NOR a usable layers_block_type must still abort
// loudly, naming the layer count — the fallback must never turn missing evidence into a
// silent 0 that runs with nonsense physics.
//
// Since #1777 the abort happens at model-config load and the diagnostic names the two
// CONFIG KEYS the resolver consults, rather than deferring to a per-backend
// "ModelConfig.NumLayers must be > 0" that names a Go struct field the operator's config
// does not contain. The loudness contract is unchanged (non-zero exit); what this asserts
// is that the operator reaches the file they have to edit.
func TestRunCmd_NoLayerCountEvidence_FailsLoudly(t *testing.T) {
	if os.Getenv("BLIS_BLOCKTYPE_NOEVIDENCE") == "1" {
		rootCmd.SetArgs([]string{
			"run",
			"--model", "nvidia/nemotron-no-layer-evidence",
			"--catalog", os.Getenv("BLIS_BLOCKTYPE_CONFIG_DIR"),
			"--hardware", "H100",
			"--hardware-config", "../hardware_config.json",
			"--tp", "1",
			"--total-kv-blocks", "20000",
			"--num-requests", "5",
			"--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
		})
		_ = rootCmd.Execute() // must logrus.Fatalf (exit 1) before returning
		os.Exit(0)
	}

	// layers_block_type present but EMPTY: the key exists, so this also guards against a
	// fallback that trusts the key's presence rather than its usable length.
	body := `{
  "layers_block_type": [],
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "vocab_size": 32000,
  "torch_dtype": "float16"
}`
	dir, err := writeTestCatalog(t.TempDir(), body, "nemotron-no-layer-evidence")
	if err != nil {
		t.Fatalf("write test catalog: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestRunCmd_NoLayerCountEvidence_FailsLoudly$")
	cmd.Env = append(os.Environ(), "BLIS_BLOCKTYPE_NOEVIDENCE=1", "BLIS_BLOCKTYPE_CONFIG_DIR="+dir)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("AC-4: a config with no usable layer-count evidence must fail loudly, got exit 0; output:\n%s", out)
	}
	for _, want := range []string{"num_hidden_layers", latency.LayersBlockTypeField} {
		if !strings.Contains(string(out), want) {
			t.Errorf("AC-4: the failure must name the config key %q the operator can set; output:\n%s", want, out)
		}
	}
	if !strings.Contains(string(out), "layer") {
		t.Errorf("AC-4: the failure must say the LAYER COUNT is what could not be determined; output:\n%s", out)
	}
}
