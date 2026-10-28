// Moved from blis-latency-kernel/packparity_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// formerRulesPin is the blis-schemas release this kernel read its rules packs from before
// they moved to blis-vllm/rules (inference-sim/inference-sim#2021 B4): the base's go.mod
// requirement, and its go.sum entry.
const (
	formerRulesPin    = "github.com/inference-sim/blis-schemas@v0.4.0"
	formerRulesPinSum = "h1:0vs4rLxdl0neSizFWCIE7A9fTm5YtYr0aSEf5jl311g="
)

// The packs this kernel reads are, byte for byte, the ones it read before they moved. The
// test downloads the former release (go mod download, which checks it against the checksum
// database and against formerRulesPinSum here) and compares each v0.29.0 and v0.31.0 pack.go
// with the one the kernel builds against today. The move rewrote exactly two things, which the
// comparison undoes and nothing else: a four-line provenance header, and the import of
// blis-schemas' internal validate package as the exported rules.Problems alias. One later
// rewrite is undone too, exactly once per pack: the capture-size check passes the engine-neutral
// rules.CUDAGraphCaptureInput built from the four fields CUDAGraphCaptureSizes reads, where it
// passed the whole engine block (blis-vllm 480d434, with blis-schemas#102 ddfd45b). Any other
// difference, one byte, fails, however the kernel's tests price.
func TestTheMovedPacksAreTheOnesThisKernelReadBefore(t *testing.T) {
	out, err := exec.Command("go", "mod", "download", "-json", formerRulesPin).Output()
	if err != nil {
		t.Fatalf("go mod download %s: %v", formerRulesPin, err)
	}
	var former struct{ Dir, Sum string }
	if err := json.Unmarshal(out, &former); err != nil {
		t.Fatal(err)
	}
	if former.Sum != formerRulesPinSum {
		t.Fatalf("%s downloaded with sum %s, want %s", formerRulesPin, former.Sum, formerRulesPinSum)
	}
	// The packs are this module's own rules/ since it folded into one module.
	out, err = exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/inference-sim/blis-vllm").Output()
	if err != nil {
		t.Fatalf("go list -m blis-vllm: %v", err)
	}
	moved := filepath.Join(strings.TrimSpace(string(out)), "rules")
	for _, pack := range []string{"v0_29", "v0_31"} {
		before, err := os.ReadFile(filepath.Join(former.Dir, "rules", pack, "pack.go"))
		if err != nil {
			t.Fatal(err)
		}
		now, err := os.ReadFile(filepath.Join(moved, pack, "pack.go"))
		if err != nil {
			t.Fatal(err)
		}
		const header = 4 // the move's provenance comment, three lines and a blank one
		lines := bytes.SplitAfter(now, []byte("\n"))
		if len(lines) <= header || !bytes.HasPrefix(lines[0], []byte("// Moved from inference-sim/blis-schemas/")) {
			t.Fatalf("%s/pack.go does not open with the move's provenance header", pack)
		}
		now = bytes.Join(lines[header:], nil)
		// Undo the capture-input rewrite: the call took the engine block before.
		const captureInput = "p.CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput{Mode: e.CUDAGraphMode,\n" +
			"\t\t\t\t\t\tMaxCaptureSize: e.MaxCUDAGraphCaptureSize, CaptureSizes: e.CUDAGraphCaptureSizes,\n" +
			"\t\t\t\t\t\tMaxNumBatchedTokens: e.MaxNumBatchedTokens})"
		if n := bytes.Count(now, []byte(captureInput)); n != 1 {
			t.Fatalf("%s/pack.go passes the capture input %d times, want once: the rewrite this test "+
				"undoes has changed, so re-verify the pack before extending it", pack, n)
		}
		now = bytes.Replace(now, []byte(captureInput), []byte("p.CUDAGraphCaptureSizes(e)"), 1)
		// Undo the engine-block rewrite (blis-vllm 0f892e3, with blis-schemas#102 2c38342): the
		// packs read a pool's block through engineconfig, decoding it in forEachPool and skipping
		// one that does not decode, which the field layer refuses (CheckEngineBlock), and the pack
		// type and its helpers moved from blis-schemas' rules to blis-vllm's pack. Every rewrite
		// is required present, so a change to one is a change this test does not undo.
		for _, r := range engineBlockRewrite {
			if !bytes.Contains(now, []byte(r.now)) {
				t.Fatalf("%s/pack.go no longer contains the engine-block rewrite %q: re-verify the "+
					"pack before changing what this test undoes", pack, r.now)
			}
			now = bytes.ReplaceAll(now, []byte(r.now), []byte(r.before))
		}
		// The one helper type only the v0.31.0 pack names (its cache-dtype resolutions).
		const resolutions = "pack.CacheDTypeResolutions"
		if pack == "v0_31" != bytes.Contains(now, []byte(resolutions)) {
			t.Fatalf("%s/pack.go names %s where the move left it otherwise", pack, resolutions)
		}
		now = bytes.ReplaceAll(now, []byte(resolutions), []byte("rules.CacheDTypeResolutions"))
		// Undo the import rewrite: drop the internal package's import line, and spell its
		// Problems type as the alias the moved pack uses.
		var kept [][]byte
		for _, l := range bytes.SplitAfter(before, []byte("\n")) {
			if !bytes.Contains(l, []byte(`"github.com/inference-sim/blis-schemas/internal/validate"`)) {
				kept = append(kept, l)
			}
		}
		before = bytes.ReplaceAll(bytes.Join(kept, nil), []byte("validate.Problems"), []byte("rules.Problems"))
		if !bytes.Equal(before, now) {
			t.Errorf("%s/pack.go differs from the pack %s shipped, beyond the move's header "+
				"and import rewrite: a rules change reached this kernel through the move, and "+
				"its pricing must be re-verified", pack, formerRulesPin)
		}
	}
}

// engineBlockRewrite is the move of the packs onto blis-vllm's engine block (0f892e3), as exact
// text: each "now" is what the moved pack holds, and "before" what it held. The decode in
// forEachPool comes first, as it spans the type names the later entries rename.
var engineBlockRewrite = []struct{ now, before string }{
	{"\t\t// A block that does not decode is the field layer's finding (Pack.CheckEngineBlock),\n" +
		"\t\t// and rules read only well-formed documents, so it is not checked here.\n" +
		"\t\te, err := engineconfig.Decode(pool.Engine)\n" +
		"\t\tif err != nil {\n" +
		"\t\t\tcontinue\n" +
		"\t\t}\n" +
		"\t\tfn(poolPath(i), engineconfig.Pool{Pool: pool, Engine: e})\n",
		"\t\tfn(poolPath(i), pool)\n"},
	{"\n\t\"github.com/inference-sim/blis-vllm/rules/engineconfig\"\n\t\"github.com/inference-sim/blis-vllm/rules/pack\"\n", ""},
	{"engineconfig.Engine", "deployment.Engine"},
	{"engineconfig.Pool", "deployment.Pool"},
	{"*pack.Pack", "*rules.Pack"},
	{"&pack.Pack{", "&rules.Pack{"},
	{"pack.SequenceParallelCondition", "rules.SequenceParallelCondition"},
	{"pack.KVCachePacking", "rules.KVCachePacking"},
	{"pack.NormalizeMoEBackend", "rules.NormalizeMoEBackend"},
}
