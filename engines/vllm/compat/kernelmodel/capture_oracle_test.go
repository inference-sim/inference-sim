package kernelmodel

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/rules"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/rules/v0_32"
)

// captureOracleCase is one row of core/lower/testdata/vllm_lower.json "capture": a deployment's
// stated batch limits, cudagraph mode and capture knobs, and what vLLM's own
// VllmConfig._set_cudagraph_sizes (vllm/config/vllm.py@5df63a72, run by oracle/lower/extract.py)
// resolves them to.
type captureOracleCase struct {
	Case struct {
		Name        string `json:"name"`
		Mode        string `json:"mode"`
		MaxSeqs     int    `json:"max_num_seqs"`
		MaxBatched  int    `json:"max_num_batched_tokens"`
		MaxCapture  *int   `json:"max_capture"`
		Sizes       []int  `json:"sizes"`
		SpecTokens  int    `json:"spec_tokens"`
		Eager       bool   `json:"enforce_eager"`
		Blackwell   bool   `json:"blackwell"`
		Performance string `json:"performance_mode"`
	} `json:"case"`
	Resolved struct {
		Max   int     `json:"max"`
		Sizes []int   `json:"sizes"`
		Error *string `json:"error"`
	} `json:"resolved"`
}

func loadCaptureOracle(t *testing.T) []captureOracleCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "core", "lower", "testdata", "vllm_lower.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Commit  string              `json:"commit"`
		Capture []captureOracleCase `json:"capture"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.Commit != "5df63a7203cdb9328d824a27b682e77c7b9928be" {
		t.Fatalf("the oracle file is for commit %s, not the pinned release", f.Commit)
	}
	return f.Capture
}

var engineLine = regexp.MustCompile(`(?m)^( +)(max_num_batched_tokens|max_num_seqs|compilation_config): .*\n`)

// openCapture opens a committed fixture at vLLM 0.32.0 on a chip with the case's batch limits,
// cudagraph mode and capture knobs written into its engine block, and returns the kernel. A
// case with speculation uses the GLM-5 MTP fixture, the rest the gpt-oss one.
func openCapture(t *testing.T, c captureOracleCase, hardware string) *latencykernel.Kernel {
	t.Helper()
	path := filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", "gpt-oss-120b-h200-fp4-vllm-tp4.yaml")
	if c.Case.SpecTokens > 0 {
		path = filepath.Join("..", "..", "testdata", "scenarios", "glm-5-h200-tp8-mtp3.yaml")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(raw), `engine_version: "0.29.0"`, `engine_version: "0.32.0"`, 1)
	if n := len(engineLine.FindAllString(body, -1)); n != 3 {
		t.Fatalf("fixture states %d of max_num_batched_tokens, max_num_seqs, compilation_config, want 3", n)
	}
	body = engineLine.ReplaceAllString(body, "")
	indent := "      "
	add := fmt.Sprintf("%smax_num_batched_tokens: %d\n%smax_num_seqs: %d\n%scompilation_config: {cudagraph_mode: %s}\n",
		indent, c.Case.MaxBatched, indent, c.Case.MaxSeqs, indent, c.Case.Mode)
	if c.Case.MaxCapture != nil {
		add += fmt.Sprintf("%smax_cudagraph_capture_size: %d\n", indent, *c.Case.MaxCapture)
	}
	if c.Case.Sizes != nil {
		sizes, _ := json.Marshal(c.Case.Sizes)
		add += fmt.Sprintf("%scudagraph_capture_sizes: %s\n", indent, sizes)
	}
	body = strings.Replace(body, "      gpu_memory_utilization:", add+"      gpu_memory_utilization:", 1)
	if c.Case.SpecTokens > 0 {
		if !strings.Contains(body, "num_speculative_tokens: 3") {
			t.Fatal("the GLM-5 MTP fixture states no num_speculative_tokens: 3 to replace")
		}
		body = strings.Replace(body, "num_speculative_tokens: 3", fmt.Sprintf("num_speculative_tokens: %d", c.Case.SpecTokens), 1)
	}
	body = strings.Replace(body, "hardware: h200", "hardware: "+hardware, 1)
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatalf("%v\n%s", err, add)
	}
	k, ok := m.Kernel().(*latencykernel.Kernel)
	if !ok {
		t.Fatalf("the kernel is a %T", m.Kernel())
	}
	return k
}

// knownCaptureDivergences are the oracle cases on which the kernel does not give vLLM's maximum,
// each with the reason. The test fails if a listed case now agrees, so the entry is removed with
// the fix. None is in the value class the dashboard's points state, because those state
// max_cudagraph_capture_size, which the kernel resolves through the rules pack.
var knownCaptureDivergences = map[string]string{
	"batched-tokens-off-stride":           unstatedMaxAboveBatchedTokens,
	"batched-tokens-off-stride-blackwell": unstatedMaxAboveBatchedTokens,
}

const unstatedMaxAboveBatchedTokens = "max_cudagraph_capture_size unstated and max_num_batched_tokens below the default ceiling: " +
	"vLLM caps the default at max_num_batched_tokens (vllm/config/vllm.py:2445-2446@5df63a72), " +
	"the kernel's unstated branch (kernel/new.go, min(seqs x width x 2, 512 or 1024)) does not, " +
	"so it prices the capture memory and the graph ceiling at 512 where vLLM has 300"

// The kernel's CUDA graph capture ceiling is vLLM's own for every configuration the oracle holds
// that an engine block can state, on a Hopper part (h200) and a data-center Blackwell part
// (b200). The value class: max_cudagraph_capture_size stated as 2048, as 2, off the stride
// (1000, which vLLM truncates to 992), equal to max_num_batched_tokens, and in [8, 256), where
// vLLM's capture stride is 8 (64, 100 which it truncates to 96, and 128), and cudagraph_mode
// and cudagraph_capture_sizes as stated; or neither stated, where the ceiling is
// min(max_num_seqs x decode width x 2, 512 or 1024 on Blackwell).
func TestKernelCaptureTokensAreVLLMs(t *testing.T) {
	cases := loadCaptureOracle(t)
	if len(cases) < 40 {
		t.Fatalf("only %d oracle capture cases", len(cases))
	}
	var checked, stated2048, stated2, off, small int
	listed := map[string]bool{}
	for _, c := range cases {
		switch {
		case c.Resolved.Error != nil:
			// vLLM refuses these at start-up; the kernel's refusals are engineconfig's, tested there.
			continue
		case c.Case.Eager:
			// enforce_eager is not an engine key; the kernel has no field to set it.
			continue
		case c.Case.MaxBatched <= 0:
			// The engine block states max_num_batched_tokens or the kernel substitutes its own default.
			continue
		case c.Case.Performance == "interactivity":
			// performance_mode is not an engine key.
			continue
		}
		hw := "h200"
		if c.Case.Blackwell {
			hw = "b200"
		}
		t.Run(c.Case.Name, func(t *testing.T) {
			k := openCapture(t, c, hw)
			got := k.CaptureTokens()
			why, known := knownCaptureDivergences[c.Case.Name]
			if known {
				listed[c.Case.Name] = true
				if got == c.Resolved.Max {
					t.Errorf("CaptureTokens %d now equals vLLM's; remove the known divergence (%s)", got, why)
				}
				return
			}
			if got != c.Resolved.Max {
				t.Errorf("CaptureTokens %d, vLLM resolves max_cudagraph_capture_size %d", got, c.Resolved.Max)
			}
			// The sizes behind the maximum, through the rules pack the kernel calls, for a
			// deployment that states the maximum or the sizes. Where it states neither, vLLM
			// also appends max_num_seqs (and, with speculation, the uniform decode sizes),
			// which the pack does not model and the kernel does not price, so only the
			// maximum is compared there.
			if c.Case.SpecTokens > 0 || c.Case.Mode == "NONE" || (c.Case.MaxCapture == nil && c.Case.Sizes == nil) {
				return
			}
			in := rules.CUDAGraphCaptureInput{Mode: c.Case.Mode, MaxNumBatchedTokens: c.Case.MaxBatched,
				MaxCaptureSize: c.Case.MaxCapture, CaptureSizes: c.Case.Sizes}
			p, _, err := v0_32.Pack().CUDAGraphCaptureSizes(in)
			if err != nil {
				t.Fatal(err)
			}
			if p.Max != c.Resolved.Max || !slices.Equal(p.Sizes, c.Resolved.Sizes) {
				t.Errorf("the v0_32 pack gives max %d sizes %v; vLLM max %d sizes %v", p.Max, p.Sizes, c.Resolved.Max, c.Resolved.Sizes)
			}
		})
		checked++
		if m := c.Case.MaxCapture; m != nil && c.Case.SpecTokens == 0 && c.Case.Sizes == nil {
			switch *m {
			case 2048:
				stated2048++
			case 2:
				stated2++
			case 1000:
				off++
			case 64, 100, 128:
				small++
			}
		}
	}
	for name := range knownCaptureDivergences {
		if !listed[name] {
			t.Errorf("known divergence %q matches no oracle case", name)
		}
	}
	// 64, 100 and 128 are three values on two parts.
	if checked < 40 || stated2048 < 2 || stated2 < 2 || off < 2 || small < 6 {
		t.Errorf("checked %d cases (2048: %d, 2: %d, 1000: %d, 64/100/128: %d); the stated values must run on both parts", checked, stated2048, stated2, off, small)
	}
}
