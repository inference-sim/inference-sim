package kernelmodel

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// TestWritePriceReport is the generator of the figures in docs/evidence/*-price-report.md (design
// X13). It runs only when BLIS_PRICE_REPORT names an output file, and writes one tab-separated row
// for each v0.32.0 pool of the committed scenario fixtures:
//
//	fixture, pool, status, blocks, bytes_per_block, block_size, cache_dtype, then the kernel's
//	NoOverlap step time in microseconds for decode at batch 1 and 256 (context 1024, one token
//	each), prefill of one 2048-token request, and decode at batch 700 (context 1024).
//
// A pool that does not open or has no KV budget has its error in the status column. Run it on the
// code before a change and after, with BLIS_PRICE_VARIANT naming a transformation of each fixture
// ("" as committed; "nocache" removes cache_dtype; "nobs" removes block_size; "nobatch-deepep" removes
// max_num_seqs and max_num_batched_tokens and states all2all_backend deepep_low_latency), and diff the files:
//
//	BLIS_PRICE_REPORT=/tmp/after.tsv BLIS_PRICE_VARIANT=nocache go test ./compat/kernelmodel -run TestWritePriceReport
func TestWritePriceReport(t *testing.T) {
	out := os.Getenv("BLIS_PRICE_REPORT")
	if out == "" {
		t.Skip("BLIS_PRICE_REPORT names the output file; the test writes the price-change report's figures")
	}
	variant := os.Getenv("BLIS_PRICE_VARIANT")
	drop := map[string]*regexp.Regexp{
		"":        nil,
		"nocache": regexp.MustCompile(`(?m)^\s*kv_cache_dtype:.*\n`),
		"nobs":    regexp.MustCompile(`(?m)^\s*block_size:.*\n`),
		// The batch-settings report's third run: both batch settings removed and
		// all2all_backend deepep_low_latency stated instead of whatever the fixture states.
		"nobatch-deepep": regexp.MustCompile(`(?m)^\s*(max_num_seqs|max_num_batched_tokens|all2all_backend):.*\n`),
	}
	re, ok := drop[variant]
	if !ok {
		t.Fatalf("BLIS_PRICE_VARIANT %q is not one of '', nocache, nobs, nobatch-deepep", variant)
	}
	var files []string
	for _, d := range []string{"kerneltests/testdata", "kerneltests/testdata/aisimulate", "kerneltests/testdata/direct", "testdata/scenarios", "vllmengine/testdata"} {
		m, _ := filepath.Glob(filepath.Join("..", "..", d, "*.yaml"))
		files = append(files, m...)
	}
	sort.Strings(files)
	release := regexp.MustCompile(`(?m)^(?:engine_version:\s*"?0\.32\.0"?|\s+version:\s*"?0\.32\.0"?)\s*$`)
	var rows []string
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		body := string(raw)
		if !release.MatchString(body) {
			continue
		}
		if re != nil {
			body = re.ReplaceAllString(body, "")
		}
		if variant == "nobatch-deepep" {
			body = regexp.MustCompile(`(?m)^(\s*)engine:\n(\s+)`).ReplaceAllString(body, "${1}engine:\n${2}all2all_backend: deepep_low_latency\n${2}")
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "v.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		r := repos()
		r.Scenarios = dir
		roles, err := RolePools("v.yaml", r)
		if err != nil {
			rows = append(rows, fmt.Sprintf("%s\t-\t%s", rel(f), oneLine(err)))
			continue
		}
		pools := map[int]bool{}
		for _, i := range roles {
			pools[i] = true
		}
		idx := make([]int, 0, len(pools))
		for i := range pools {
			idx = append(idx, i)
		}
		sort.Ints(idx)
		for _, i := range idx {
			rows = append(rows, priceRow(rel(f), i, r))
		}
	}
	if err := os.WriteFile(out, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %d rows to %s", len(rows), out)
}

func rel(f string) string {
	f = filepath.ToSlash(f)
	return strings.TrimPrefix(f, "../../")
}

func oneLine(err error) string {
	s := strings.Join(strings.Fields(err.Error()), " ")
	if len(s) > 160 {
		s = s[:160]
	}
	return s
}

func priceRow(fixture string, pool int, r Repos) string {
	m, err := OpenPool("v.yaml", r, pool)
	if err != nil {
		return fmt.Sprintf("%s\t%d\t%s", fixture, pool, oneLine(err))
	}
	e, err := m.Engine()
	if err != nil {
		return fmt.Sprintf("%s\t%d\t%s", fixture, pool, oneLine(err))
	}
	b, err := m.KVBudget()
	status, blocks, per := "ok", int64(0), int64(0)
	if err != nil {
		status = oneLine(err)
	} else {
		blocks, per = b.PerRankBlocks, b.PerBlockBytes
	}
	// The kernel's NoOverlap time, unrounded, in microseconds.
	us := func(shapes []ReqShape) string {
		d := m.k.StepTime(kernel.Batch{Reqs: shapes, DecodeThreshold: m.decodeThreshold, SMBudget: m.smBudget}).NoOverlap
		return fmt.Sprintf("%.3f", float64(d)/1e3)
	}
	decode := func(n int) []ReqShape {
		s := make([]ReqShape, n)
		for i := range s {
			s[i] = ReqShape{Scheduled: 1, Computed: 1023, PromptLen: 512}
		}
		return s
	}
	prefill := []ReqShape{{Scheduled: 2048, Computed: 0, PromptLen: 2048}}
	return strings.Join([]string{fixture, fmt.Sprint(pool), status, fmt.Sprint(blocks), fmt.Sprint(per), fmt.Sprint(e.BlockSize),
		e.CacheDType, us(decode(1)), us(decode(256)), us(prefill), us(decode(700))}, "\t")
}
