package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
)

// Config-time rank coupling: --lora-instance-max-rank and --lora-instance-capacity.

func TestParseLoRAInstanceList(t *testing.T) {
	ok := []struct {
		in   string
		want []int
	}{
		{"", nil},
		{"   ", nil},
		{"8", []int{8}},
		{"8,64,16", []int{8, 64, 16}},
		{" 8 , 64 ", []int{8, 64}},
	}
	for _, tc := range ok {
		got, err := parseLoRAInstanceList(tc.in)
		if err != nil || !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parseLoRAInstanceList(%q) = %v, %v; want %v, nil", tc.in, got, err, tc.want)
		}
	}
	// An empty element must be an error, never skipped: skipping would shift every
	// later instance's value onto its neighbour.
	bad := []struct{ in, want string }{
		{"8,,64", "element 1 is empty"},
		{"8,64,", "element 2 is empty"},
		{",8", "element 0 is empty"},
		{"8,6.4e1", `element 1 "6.4e1" is not an integer`},
		{"8,true", `element 1 "true" is not an integer`},
		{"8;64", `element 0 "8;64" is not an integer`},
	}
	for _, tc := range bad {
		_, err := parseLoRAInstanceList(tc.in)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("parseLoRAInstanceList(%q) error = %v, want one containing %q", tc.in, err, tc.want)
		}
	}
}

// instanceConfigRun re-executes this test binary as `blis run` on two instances with
// the per-instance fixture, plus extra flags from BLIS_INSTCFG_EXTRA (space-separated),
// and returns stdout, stderr and the exit code.
func instanceConfigRun(t *testing.T, extra ...string) (stdout, stderr string, code int) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestLoRAInstanceConfigCLI_Subprocess")
	cmd.Env = append(os.Environ(), "BLIS_INSTCFG_SUBPROCESS=1", "BLIS_INSTCFG_EXTRA="+strings.Join(extra, " "))
	var out, errOut bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errOut
	err := cmd.Run()
	var exitErr *exec.ExitError
	switch {
	case err == nil:
		code = 0
	case errors.As(err, &exitErr):
		code = exitErr.ExitCode()
	default:
		t.Fatalf("subprocess did not start: %v", err)
	}
	return out.String(), errOut.String(), code
}

// TestLoRAInstanceConfigCLI_Subprocess is the child half of instanceConfigRun; it does
// nothing when run directly.
func TestLoRAInstanceConfigCLI_Subprocess(t *testing.T) {
	if os.Getenv("BLIS_INSTCFG_SUBPROCESS") != "1" {
		t.Skip("child half of instanceConfigRun")
	}
	args := []string{
		"run", "--model", "qwen/qwen3-14b", "--hardware", "H100", "--tp", "1", "--seed", "42",
		"--catalog", "../testdata/catalog", "--defaults-filepath", "../defaults.yaml",
		"--num-instances", "2", "--lora-config", "testdata/lora_instance_config.yaml",
		"--creation-policy", "pre-placement", "--lora-adapter-placement", "0=a8;1=c64",
		"--routing-policy", "route-to-holder",
	}
	if extra := os.Getenv("BLIS_INSTCFG_EXTRA"); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}
	rootCmd.SetArgs(args)
	_ = rootCmd.Execute()
	os.Exit(0)
}

// The echo reaches the output with each instance's configuration, its reservation equals
// capacity × footprint × rank computed here, and the larger reservation leaves fewer KV
// blocks. Without the flags the key is absent (INV-6).
func TestLoRAInstanceConfigCLI_EchoesConfigurationThatRan(t *testing.T) {
	// Read the --metrics-path file, which is what the harness consumes; with two
	// instances stdout also carries per-instance blocks.
	metrics := filepath.Join(t.TempDir(), "metrics.json")
	_, stderr, code := instanceConfigRun(t, "--lora-instance-max-rank", "8,64", "--lora-instance-capacity", "1,1",
		"--metrics-path", metrics)
	if code != 0 {
		t.Fatalf("run exited %d; stderr:\n%s", code, stderr)
	}
	body, err := os.ReadFile(metrics)
	if err != nil {
		t.Fatalf("read metrics file: %v", err)
	}
	var got struct {
		LoRAInstances []sim.LoRAInstanceEcho `json:"lora_instances"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if len(got.LoRAInstances) != 2 {
		t.Fatalf("lora_instances has %d entries, want 2:\n%s", len(got.LoRAInstances), body)
	}
	const footprint = 2.0e6 // testdata/lora_instance_config.yaml
	for i, rank := range []int{8, 64} {
		e := got.LoRAInstances[i]
		want := int64(1 * float64(rank) * footprint)
		if e.InstanceID != fmt.Sprintf("instance_%d", i) || e.MaxLoRARank != rank || e.AdapterCapacity != 1 {
			t.Errorf("echo %d = %+v, want instance_%d at rank %d, capacity 1", i, e, i, rank)
		}
		if e.AdapterReservedBytes != want {
			t.Errorf("echo %d reservation = %d, want 1 × %d × %.0f = %d", i, e.AdapterReservedBytes, rank, footprint, want)
		}
	}
	if got.LoRAInstances[0].TotalKVBlocks <= got.LoRAInstances[1].TotalKVBlocks {
		t.Errorf("rank-8 instance has %d KV blocks, rank-64 instance %d; the smaller reservation must leave more",
			got.LoRAInstances[0].TotalKVBlocks, got.LoRAInstances[1].TotalKVBlocks)
	}

	plain, stderr, code := instanceConfigRun(t)
	if code != 0 {
		t.Fatalf("run without the flags exited %d; stderr:\n%s", code, stderr)
	}
	if strings.Contains(plain, "lora_instances") {
		t.Errorf("lora_instances present without the flags (INV-6)")
	}
}

// A violation exits through logrus.Fatalf (exit 1, a message) rather than
// NewClusterSimulator's panic (exit 2, a stack), for a CLI-shape error and for a
// placement-against-cap error alike.
func TestLoRAInstanceConfigCLI_ViolationsAreFatal(t *testing.T) {
	cases := []struct {
		name  string
		extra []string
		want  string
	}{
		{"seeded rank above cap", []string{"--lora-instance-max-rank", "8,32", "--lora-instance-capacity", "1,1"},
			`instance 1 assigned adapter \"c64\" of rank 64, above its max_lora_rank 32`},
		{"malformed list", []string{"--lora-instance-max-rank", "8,,64", "--lora-instance-capacity", "1,1"},
			"Invalid --lora-instance-max-rank: element 1 is empty"},
		{"explicit total-kv-blocks", []string{"--lora-instance-max-rank", "8,64", "--lora-instance-capacity", "1,1", "--total-kv-blocks", "5000"},
			"requires auto-calculated KV capacity"},
		// 100 × 512 × 2e6 bytes is 102 GB, beyond an 80 GiB H100: a sizing failure, which
		// validation now reports instead of the construction-time panic.
		{"reservation larger than the GPU", []string{"--lora-instance-max-rank", "512,64", "--lora-instance-capacity", "100,1"},
			"instance 0: per-instance KV sizing with max_lora_rank=512"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, stderr, code := instanceConfigRun(t, tc.extra...)
			if code != 1 {
				t.Fatalf("exit code %d, want 1 (logrus.Fatalf); stderr:\n%s", code, stderr)
			}
			if !strings.Contains(stderr, tc.want) {
				t.Errorf("stderr lacks %q:\n%s", tc.want, stderr)
			}
		})
	}
}

// replay has no per-instance KV auto-calc, so it refuses both flags rather than
// replaying a different cluster (INV-13). Mirrors TestReplayCmd_AutoscalerFlagFatal.
func TestReplayCmd_LoRAInstanceFlagsFatal(t *testing.T) {
	for _, flag := range []string{"lora-instance-max-rank", "lora-instance-capacity"} {
		t.Run(flag, func(t *testing.T) {
			if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
				if os.Getenv("BLIS_INSTCFG_FLAG") != flag {
					t.Skip("other subtest's child")
				}
				dir := t.TempDir()
				headerPath := filepath.Join(dir, "trace.yaml")
				dataPath := filepath.Join(dir, "trace.csv")
				_ = os.WriteFile(headerPath, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0644)
				_ = os.WriteFile(dataPath, []byte("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"), 0644)
				catalogDir, hwPath := setupTrainedPhysicsTestFixtures(t)
				testCmd := &cobra.Command{}
				registerSimConfigFlags(testCmd)
				testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
				testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
				if err := testCmd.ParseFlags([]string{
					"--model", "test-model", "--latency-model", "trained-physics",
					"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
					"--catalog", catalogDir, "--hardware-config", hwPath,
					"--trace-header", headerPath, "--trace-data", dataPath,
					"--" + flag, "8",
					"--defaults-filepath", "../defaults.yaml",
				}); err != nil {
					fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
					os.Exit(2)
				}
				replayCmd.Run(testCmd, nil) // must Fatalf before here
				os.Exit(0)
			}
			cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_LoRAInstanceFlagsFatal/"+flag, "-test.v")
			cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1", "BLIS_INSTCFG_FLAG="+flag)
			out, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
				t.Fatalf("want exit 1 (logrus.Fatalf), got %v; output:\n%s", err, out)
			}
			if want := "--" + flag + " is not supported in blis replay"; !strings.Contains(string(out), want) {
				t.Errorf("output lacks %q:\n%s", want, out)
			}
		})
	}
}

// When KV cannot hold one full max_position_embeddings sequence, the global auto-calc caps
// max-model-len from the cluster-wide reservation. Each instance must be re-capped from the
// uncapped value by its own budget instead: at --gpu-memory-utilization 0.45 the global cap
// fires (asserted from stderr, the premise), and the rank-8 and rank-64 instances, whose
// budgets differ, must then report different max_model_len, each its own blocks × 16.
func TestLoRAInstanceConfigCLI_MaxModelLenPerInstance(t *testing.T) {
	metrics := filepath.Join(t.TempDir(), "metrics.json")
	_, stderr, code := instanceConfigRun(t, "--lora-instance-max-rank", "8,64", "--lora-instance-capacity", "1,1",
		"--gpu-memory-utilization", "0.45", "--metrics-path", metrics)
	if code != 0 {
		t.Fatalf("run exited %d; stderr:\n%s", code, stderr)
	}
	if !strings.Contains(stderr, "capping to") {
		t.Fatalf("premise: the global max-model-len cap did not fire; stderr:\n%s", stderr)
	}
	body, err := os.ReadFile(metrics)
	if err != nil {
		t.Fatalf("read metrics file: %v", err)
	}
	var got struct {
		LoRAInstances []sim.LoRAInstanceEcho `json:"lora_instances"`
	}
	if err := json.Unmarshal(body, &got); err != nil || len(got.LoRAInstances) != 2 {
		t.Fatalf("decode lora_instances: %v (%d entries)", err, len(got.LoRAInstances))
	}
	for i, e := range got.LoRAInstances {
		if want := e.TotalKVBlocks * 16; e.MaxModelLen != want {
			t.Errorf("instance %d max_model_len = %d, want its own KV-feasible %d (blocks %d × 16)", i, e.MaxModelLen, want, e.TotalKVBlocks)
		}
	}
	if got.LoRAInstances[0].MaxModelLen == got.LoRAInstances[1].MaxModelLen {
		t.Errorf("both instances run at max_model_len %d, the global cap", got.LoRAInstances[0].MaxModelLen)
	}
}
