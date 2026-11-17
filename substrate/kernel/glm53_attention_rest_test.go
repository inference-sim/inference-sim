package latencykernel

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/load"
)

// glm53RestFile is the generated file registry/scripts/fit_glm53_attention.py fits against: for
// every prefill row of AISimulate's GLM-5.3 attention-module table at TP8, what this kernel
// charges a layer for the module EXCEPT its sparse-MLA prefill attention, the part the fitted
// form prices. The fit is the measured module minus this, so the file is regenerated whenever
// the kernel's price of the rest of the module moves, and the fit is redone.
//
// Regenerate with
//
//	BLIS_WRITE_GLM53_REST=1 go test ./kernel -run TestGLM53AttentionRestIsCurrent
//
// then re-run the writer (python scripts/fit_glm53_attention.py, from registry/).
const glm53RestFile = "../registry/docs/glm53-attention-rest-h200-tp8.csv"

const glm53RestHeader = "batch,new_tokens,prefix,rest_us"

// glm53ModuleProbe is glm-5.3-flash's sparse-MLA layer cut before its attention-output
// all-reduce (the module AISimulate times: projections, the indexer and its scorer, the
// sparse attention and o_proj; the table's collection_meta.yaml records the output
// all-reduce as excluded), stacked n deep, with no speculator. Two depths are priced and
// differenced so the per-step costs that do not scale with layers (head, embedding) cancel.
func glm53ModuleProbe(t *testing.T, g *model.Graph, n int) *model.Graph {
	t.Helper()
	var kind *model.LayerKind
	for i := range g.LayerKinds {
		if g.LayerKinds[i].ID == "mla_moe" {
			kind = &g.LayerKinds[i]
		}
	}
	if kind == nil {
		t.Fatal("glm-5.3-flash has no mla_moe layer kind")
	}
	cut := slices.IndexFunc(kind.Nodes, func(nd model.Node) bool { return nd.Role == "attn_out" })
	if cut < 0 {
		t.Fatal("glm-5.3-flash's mla_moe layer has no attn_out node")
	}
	k := model.LayerKind{ID: "mla_attn", Nodes: slices.Clone(kind.Nodes[:cut])}
	for _, e := range kind.Edges {
		if e[0] < cut && e[1] < cut {
			k.Edges = append(k.Edges, e)
		}
	}
	p := *g
	p.LayerKinds = []model.LayerKind{k}
	p.Stack = model.Stack{Pattern: []string{k.ID}, Repeat: n}
	p.Speculator = nil
	return &p
}

// withoutSparsePrefillCost is in with the sparse-MLA prefill form for glm-5.3-flash on h200
// charging nothing measurable, every other entry of those names removed: the step then prices
// everything in the layer except that form.
func withoutSparsePrefillCost(in Inputs) Inputs {
	values := map[string]float64{
		"attention_prefill_floor_sparse_mla": 1e-12, "attention_prefill_per_request_sparse_mla": 1e-12,
		"attention_prefill_per_token_sparse_mla": 1e-12, "attention_prefill_rate_sparse_mla": 1e30,
	}
	units := map[string]string{
		"attention_prefill_floor_sparse_mla": "us_per_transfer", "attention_prefill_per_request_sparse_mla": "us_per_request",
		"attention_prefill_per_token_sparse_mla": "us_per_token", "attention_prefill_rate_sparse_mla": "bytes_per_us",
	}
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if _, ours := values[e.Name]; !ours {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		if c.Name == "cost-model-attention" {
			for _, name := range slices.Sorted(maps.Keys(values)) {
				c.Coefficients = append(c.Coefficients, coefficient.Entry{
					Name: name, Value: values[name], Units: vocab.Unit(units[name]), Method: "measured",
					Scope: coefficient.Scope{Hardware: []string{"h200"}, Model: []string{"glm-5.3-flash"}},
				})
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// glm53TableCase is the plan blis-vllm lowers the table's cell to (see TestGLM53AttentionRestIsCurrent).
func glm53TableCase(t *testing.T, d *load.Data) Inputs {
	t.Helper()
	return planCase(t, d, "glm-5.3-flash", "h200", 1, "tp: 8, pp: 1, dp: 1", func(l *plan.Lowered) {
		l.Plan.MaxContext, l.Plan.PageTokens, l.Plan.MemoryFraction = 262144, 64, 0.9
		l.Plan.Batch = plan.BatchLimits{Requests: 256, Tokens: 65536}
		l.Plan.WeightFormat = "fp8"
		l.Plan.Cache = plan.CacheFormat{Record: "auto", ElementBytes: 2, StateElementBytes: 4}
		l.Plan.Engram = plan.EngramPlacement{Host: true, Shards: 8}
		buckets := []int{1, 2, 4}
		for b := 8; b <= 256; b += 8 {
			buckets = append(buckets, b)
		}
		for b := 272; b <= 512; b += 16 {
			buckets = append(buckets, b)
		}
		l.Plan.Graphs = plan.GraphPolicy{Decode: plan.LaunchGraph, Other: plan.LaunchPiecewise,
			Ceiling: 512, Buckets: buckets}
	})
}

type glm53RestRow struct {
	batch, newTokens, prefix int
	restUS                   float64
}

func readGLM53Rest(t *testing.T) []glm53RestRow {
	t.Helper()
	raw, err := os.ReadFile(glm53RestFile)
	if err != nil {
		t.Fatal(err)
	}
	recs, err := csv.NewReader(bytes.NewReader(raw)).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) < 2 || strings.Join(recs[0], ",") != glm53RestHeader {
		t.Fatalf("%s: header is not %q", glm53RestFile, glm53RestHeader)
	}
	var rows []glm53RestRow
	for _, r := range recs[1:] {
		var v [3]int
		for i := range v {
			if v[i], err = strconv.Atoi(r[i]); err != nil {
				t.Fatalf("%s: %v", glm53RestFile, err)
			}
		}
		us, err := strconv.ParseFloat(r[3], 64)
		if err != nil {
			t.Fatalf("%s: %v", glm53RestFile, err)
		}
		rows = append(rows, glm53RestRow{v[0], v[1], v[2], us})
	}
	return rows
}

// The committed rest-of-module prices are what this kernel charges today. When a change to the
// kernel or to any coefficient the module reads moves them, this fails, and the file and the
// glm-5.3 sparse-MLA prefill fit that subtracts it must be regenerated together. The rederive
// job runs it beside the writers' --check (registry/scripts/rederive.sh).
func TestGLM53AttentionRestIsCurrent(t *testing.T) {
	t.Run("decode", testGLM53DecodeRestIsCurrent)
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	// The table's measurement: TP8, no expert parallelism, a BF16 KV cache in 64-token pages.
	// The plan is the one blis-vllm (e7855679, vLLM v0.32.0's rules) lowers that cell to
	// (TP8, fp8 checkpoint, kv_cache_dtype auto, max_num_seqs 256, max_num_batched_tokens
	// 8192, max_model_len 262144): full graphs for decode and piecewise graphs for the rest up
	// to 512 tokens. Its token budget is raised past 8,192 so a batch-2 x 4,096 row is one
	// step, as the table timed it.
	in := glm53TableCase(t, d)
	in = withoutSparsePrefillCost(in)
	in.Draft = nil
	price := func(n int) func(r glm53RestRow) float64 {
		probe := in
		probe.Model = glm53ModuleProbe(t, in.Model, n)
		k := mustKernel(t, probe)
		return func(r glm53RestRow) float64 {
			reqs := make([]kernel.ReqShape, r.batch)
			for i := range reqs {
				reqs[i] = kernel.ReqShape{Scheduled: r.newTokens, Computed: r.prefix,
					PromptLen: r.prefix + r.newTokens}
			}
			// The table is GPU kernel time only (CUPTI busy time over graph replay), so the
			// rest is the device's share of the price: the SM and HBM resources, not the
			// host's issue time.
			e := k.StepTime(kernel.Batch{Reqs: reqs, DecodeThreshold: 1})
			return (e.PerResource[kernel.ResourceSM] + e.PerResource[kernel.ResourceHBM]).Seconds() * 1e6
		}
	}
	p10, p20 := price(10), price(20)
	rows := readGLM53Rest(t)
	got := make([]float64, len(rows))
	for i, r := range rows {
		got[i] = (p20(r) - p10(r)) / 10
		if !(got[i] > 0) {
			t.Fatalf("row %d,%d,%d: the rest of the module prices at %v us a layer", r.batch,
				r.newTokens, r.prefix, got[i])
		}
	}
	if os.Getenv("BLIS_WRITE_GLM53_REST") == "1" {
		var b strings.Builder
		b.WriteString(glm53RestHeader + "\n")
		for i, r := range rows {
			fmt.Fprintf(&b, "%d,%d,%d,%.4f\n", r.batch, r.newTokens, r.prefix, got[i])
		}
		if err := os.WriteFile(glm53RestFile, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	bad := 0
	for i, r := range rows {
		// Four decimals are written; the tolerance also absorbs a fused multiply-add on
		// another architecture.
		if math.Abs(got[i]-r.restUS) > 1e-4+1e-7*r.restUS {
			if bad < 5 {
				t.Errorf("row %d,%d,%d: committed %.4f us, the kernel now charges %.4f us", r.batch,
					r.newTokens, r.prefix, r.restUS, got[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d rows of %s are stale: regenerate it (BLIS_WRITE_GLM53_REST=1) and re-run "+
			"registry/scripts/fit_glm53_attention.py, which fits against it", bad, len(rows), glm53RestFile)
	}
}

// glm53DecodeRestFile is the decode counterpart of glm53RestFile: for every decode row of the
// same table at TP8 (one query token a request, the row's x as each request's context), what this
// kernel charges a layer for the module EXCEPT the sparse-MLA decode attention, and the bytes that
// attention reads as the kernel counts them (decodeReadBytes). registry/scripts/
// fit_glm53_attention.py fits the decode floor and rate against it.
//
// Unlike the prefill file, the rest here INCLUDES the probe's kernel launches at
// host_launch_per_kernel (its committed value, read from the registry). The table is GPU busy time,
// and this kernel charges every launch on top of the device terms, so a decode form fitted against
// the device share alone would charge the module's launches twice; at decode they are about a
// third of the module, at prefill under 2% of a row, which is why the prefill file leaves them out.
// The per-step host cost (graph replay) cancels in the two-depth difference.
//
// Regenerate with BLIS_WRITE_GLM53_REST=1, as the prefill file (TestGLM53AttentionRestIsCurrent/decode).
const glm53DecodeRestFile = "../registry/docs/glm53-attention-decode-rest-h200-tp8.csv"

const glm53DecodeRestHeader = "batch,context,rest_us,read_bytes"

// glm53DecodeProbeRate is the decode rate, bytes a microsecond, the read probe prices at: the
// attention's share of the step is then its read in thousands of bytes.
const glm53DecodeProbeRate = 1e3

// withSparseDecode is in with glm-5.3-flash's sparse-MLA decode pair on h200 set to floor and rate,
// every other entry of those names removed.
func withSparseDecode(in Inputs, floor, rate float64) Inputs {
	values := map[string]float64{"attention_decode_floor_sparse_mla": floor, "attention_decode_rate_sparse_mla": rate}
	units := map[string]string{"attention_decode_floor_sparse_mla": "us_per_transfer",
		"attention_decode_rate_sparse_mla": "bytes_per_us"}
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if _, ours := values[e.Name]; !ours {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		if c.Name == "cost-model-attention" {
			for _, name := range slices.Sorted(maps.Keys(values)) {
				c.Coefficients = append(c.Coefficients, coefficient.Entry{
					Name: name, Value: values[name], Units: vocab.Unit(units[name]), Method: "measured",
					Scope: coefficient.Scope{Hardware: []string{"h200"}, Model: []string{"glm-5.3-flash"}},
				})
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

type glm53DecodeRow struct {
	batch, context int
	restUS, readB  float64
}

func readGLM53DecodeRest(t *testing.T) []glm53DecodeRow {
	t.Helper()
	raw, err := os.ReadFile(glm53DecodeRestFile)
	if err != nil {
		t.Fatal(err)
	}
	recs, err := csv.NewReader(bytes.NewReader(raw)).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) < 2 || strings.Join(recs[0], ",") != glm53DecodeRestHeader {
		t.Fatalf("%s: header is not %q", glm53DecodeRestFile, glm53DecodeRestHeader)
	}
	var rows []glm53DecodeRow
	for _, r := range recs[1:] {
		b, e1 := strconv.Atoi(r[0])
		x, e2 := strconv.Atoi(r[1])
		us, e3 := strconv.ParseFloat(r[2], 64)
		rb, e4 := strconv.ParseFloat(r[3], 64)
		if err := errors.Join(e1, e2, e3, e4); err != nil {
			t.Fatalf("%s: %v", glm53DecodeRestFile, err)
		}
		rows = append(rows, glm53DecodeRow{b, x, us, rb})
	}
	return rows
}

// testGLM53DecodeRestIsCurrent checks the committed decode rest and read bytes are what this kernel
// charges today: the same contract as the prefill file, for the decode fit. It runs as a subtest of
// TestGLM53AttentionRestIsCurrent, which the rederive job runs.
func testGLM53DecodeRestIsCurrent(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := glm53TableCase(t, d)
	in.Draft = nil
	price := func(in Inputs, n int) func(b, x int) float64 {
		probe := in
		probe.Model = glm53ModuleProbe(t, in.Model, n)
		k := mustKernel(t, probe)
		return func(b, x int) float64 {
			reqs := make([]kernel.ReqShape, b)
			for i := range reqs {
				// The table's decode rows keep their true sequence length x: x-1 computed, one
				// token scheduled (harness.DecodeBatch's shape).
				reqs[i] = kernel.ReqShape{Scheduled: 1, Computed: x - 1, PromptLen: x}
			}
			return k.StepTime(kernel.Batch{Reqs: reqs, DecodeThreshold: 1}).NoOverlap.Seconds() * 1e6
		}
	}
	zero := withSparseDecode(in, 1e-12, 1e30)
	probe := withSparseDecode(in, 1e-12, glm53DecodeProbeRate)
	r10, r20 := price(zero, 10), price(zero, 20)
	q10, q20 := price(probe, 10), price(probe, 20)
	rows := readGLM53DecodeRest(t)
	rest := make([]float64, len(rows))
	read := make([]float64, len(rows))
	for i, r := range rows {
		rest[i] = (r20(r.batch, r.context) - r10(r.batch, r.context)) / 10
		read[i] = ((q20(r.batch, r.context)-q10(r.batch, r.context))/10 - rest[i]) * glm53DecodeProbeRate
		if !(rest[i] > 0) || !(read[i] > 0) {
			t.Fatalf("row %d,%d: rest %v us, read %v B a layer", r.batch, r.context, rest[i], read[i])
		}
	}
	if os.Getenv("BLIS_WRITE_GLM53_REST") == "1" {
		var b strings.Builder
		b.WriteString(glm53DecodeRestHeader + "\n")
		for i, r := range rows {
			fmt.Fprintf(&b, "%d,%d,%.4f,%.1f\n", r.batch, r.context, rest[i], read[i])
		}
		if err := os.WriteFile(glm53DecodeRestFile, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	bad := 0
	for i, r := range rows {
		if math.Abs(rest[i]-r.restUS) > 1e-4+1e-7*r.restUS || math.Abs(read[i]-r.readB) > 0.1+1e-7*r.readB {
			if bad < 5 {
				t.Errorf("row %d,%d: committed %.4f us / %.1f B, the kernel now charges %.4f us / %.1f B",
					r.batch, r.context, r.restUS, r.readB, rest[i], read[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d rows of %s are stale: regenerate it (BLIS_WRITE_GLM53_REST=1) and re-run "+
			"registry/scripts/fit_glm53_attention.py, which fits against it", bad, len(rows), glm53DecodeRestFile)
	}
}

// keyedSparseDecode reports, as the pool's attention.decode implementations, the builds that the
// registry's transferred sparse-MLA decode entries for this scenario's model and chip are keyed to
// (the engine pin's FLASHINFER_MLA_SPARSE_SM90 decode build), so a lookup of those entries
// matches, as it does where the engine release lists that build (VAL-8). It returns false when no
// such entry scopes this model on this chip. The perturbation prices these cases, the way
// keyedLane gives it the trtllm-gen lane.
func keyedSparseDecode(in Inputs) (Inputs, bool) {
	if in.Model == nil || in.Chip == nil {
		return in, false
	}
	var impls []plan.KernelImpl
	for _, s := range in.Coefficients {
		for _, e := range s.Coefficients {
			if (e.Name != "attention_decode_floor_sparse_mla" && e.Name != "attention_decode_rate_sparse_mla") ||
				e.Method != vocab.MethodTransferred || len(e.KernelImpls) == 0 ||
				!slices.Contains(e.Scope.Hardware, in.Chip.Name) || !slices.Contains(e.Scope.Model, in.Model.Name) {
				continue
			}
			for _, k := range e.KernelImpls {
				impl := plan.KernelImpl{Library: k.Library, Variant: k.Variant, CodeHash: k.CodeHash}
				if !slices.Contains(impls, impl) {
					impls = append(impls, impl)
				}
			}
		}
	}
	if len(impls) == 0 {
		return in, false
	}
	return withRunning(in, plan.PrimAttentionDecode, impls...), true
}
