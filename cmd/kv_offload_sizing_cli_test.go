package cmd

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The build site sizes the offload tier (build.SizeOffloadBlocks): a run refuses a CPU tier
// smaller than one offload chunk with the engine's message and a non-zero exit, and a replay keeps
// the per-block bytes its trace header records (INV-13) instead of re-deriving them from the
// kernel. The header's bytes are doubled before the replay, so a replay that re-derived them
// would re-export the kernel's value.
func TestKVOffload_TheBuildSiteSizesTheBlockAndAReplayKeepsTheHeadersBytes(t *testing.T) {
	const scenario = "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"
	common := []string{"--scenario", scenario, "--seed", "7", "--horizon", "600000000"}
	_, stderr, err := runKernelCLI(t, append([]string{"run", "--num-requests", "4", "--rate", "4",
		"--kv-offload-config", writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 1024\n")}, common...)...)
	if err == nil || !strings.Contains(stderr, "kv_offload: cpu_bytes_to_use (1024) is smaller than one offload chunk") {
		t.Errorf("a CPU tier smaller than one chunk was not refused: %v\n%s", err, lastLines(stderr, 3))
	}

	perBlock := regexp.MustCompile(`per_block_bytes: (\d+)`)
	dir := t.TempDir()
	run, rerun := filepath.Join(dir, "run"), filepath.Join(dir, "replay")
	if _, stderr, err := runKernelCLI(t, append([]string{"run", "--num-requests", "20", "--rate", "15", "--trace-output", run,
		"--kv-offload-config", writeOffloadYAML(t, "kv_offload:\n  cpu_bytes_to_use: 4294967296\n")}, common...)...); err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	header := readFile(t, run+".yaml")
	m := perBlock.FindStringSubmatch(header)
	if m == nil {
		t.Fatalf("the run's trace header records no per_block_bytes:\n%s", header)
	}
	derived, _ := strconv.ParseInt(m[1], 10, 64)
	if derived <= 0 {
		t.Fatalf("the run recorded %d bytes per block", derived)
	}
	stated := strconv.FormatInt(2*derived, 10)
	if err := os.WriteFile(run+".yaml", []byte(strings.Replace(header, m[0], "per_block_bytes: "+stated, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", run + ".yaml", "--trace-data", run + ".csv",
		"--trace-output", rerun}, common...)...); err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	if got := perBlock.FindStringSubmatch(readFile(t, rerun+".yaml")); got == nil || got[1] != stated {
		t.Errorf("the replay re-exported per_block_bytes %v; its header stated %s (the kernel derives %d)", got, stated, derived)
	}
}
