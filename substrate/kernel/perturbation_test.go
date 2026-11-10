package latencykernel

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
	"gopkg.in/yaml.v3"
)

// pricesOf is everything a caller can ask a kernel for, as one comparable string: the memory
// the model occupies, what a sequence holds, three steps, the host overheads, and a handoff
// to another node. A coefficient that changes none of it changes no price.
func pricesOf(k *Kernel) string {
	var b strings.Builder
	f := k.FixedBytes()
	fmt.Fprintf(&b, "mem %d %d %d %d %d|seq %d %d|", f.Weights, f.ActivationPeak, f.CUDAGraph, f.EPLBRedundant, f.CommBuffers,
		k.SequenceFixedBytes(), k.SequenceVariableBytes(4096))
	decode := kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096}
	prefill := kernel.ReqShape{Scheduled: 512, PromptLen: 512}
	// A prefill of the engine's whole token budget, which is where a collective's message is
	// large enough for its bandwidth ceiling (the peak rate) to bind rather than its ramp.
	huge := kernel.ReqShape{Scheduled: 8192, PromptLen: 8192}
	for _, reqs := range [][]kernel.ReqShape{{decode}, {prefill}, {decode, prefill, decode}, {huge}} {
		e := k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: reqs})
		fmt.Fprintf(&b, "step %d %d|", e.Overlap, e.NoOverlap)
	}
	fmt.Fprintf(&b, "host %d %d %d|", k.AdmissionOverhead(1024), k.OutputTokenOverhead(), k.CompletionOverhead())
	fmt.Fprintf(&b, "pd %d", k.PDTransferTime(4096, kernel.Placement{Node: 0, Pool: deployment.RolePrefill}, kernel.Placement{Node: 1, Pool: deployment.RoleDecode}))
	return b.String()
}

// perturbationCase is one buildable (model, chip, layout, engine settings) the perturbation prices.
type perturbationCase struct {
	label string
	in    Inputs
}

// perturbationScenarios opens every catalog model on every catalog chip at tensor-parallel
// widths 2, 4 and 8, and at the block sizes and cache dtypes the engine accepts, keeping each
// combination the cost model builds.
func perturbationScenarios(t *testing.T, d *load.Data) []perturbationCase {
	t.Helper()
	sets := "[cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]"
	var out []perturbationCase
	for _, m := range d.Models() {
		for _, c := range d.Chips() {
			ch, _ := d.Chip(c)
			// Layouts: tensor parallel alone at 2, 4 and 8; and expert parallel over data parallel
			// ranks, which is where all-to-all, reduce-scatter and the expert all-gathers run.
			for _, lay := range perturbationLayouts {
				tp, dp, ep := lay.tp, lay.dp, lay.ep
				built := false
				for _, st := range perturbationSettings(lay) {
					st.MaxNumSeqs, st.MaxNumBatchedTokens, st.MaxModelLen = 64, 8192, 8192
					perNode := 8
					if ch != nil && ch.GPUsPerNode > 0 {
						perNode = ch.GPUsPerNode
					}
					nodes := (tp*dp + perNode - 1) / perNode
					packaging := fmt.Sprintf("  gpus_per_node: %d\n", perNode)
					if ch != nil && ch.GPUsPerNode > 0 {
						packaging = ""
					}
					body := fmt.Sprintf("kind: Scenario\nname: p\nmodel: %s\ncoefficients: %s\nengine_version: \"0.32.0\"\ncluster:\n  hardware: %s\n  fabric: ib-400g\n  nodes: %d\n%s---\nkind: Deployment\nname: d\npools:\n  - role: colocated\n    nodes: %d\n    parallel: {tp: %d, pp: 1, dp: %d, enable_expert_parallel: %t%s}\n", m, sets, c, nodes, packaging, nodes, tp, dp, ep, ctxPar(lay))
					dir := t.TempDir()
					if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(body), 0o644); err != nil {
						t.Fatal(err)
					}
					stCopy := st
					r := Repos{Scenarios: dir, Data: d, Rules: func(string) rules.EngineRules { return windowedRules{} },
						EngineSettings: func(*deployment.Deployment, int) (enginesettings.Settings, error) { return stCopy, nil }}
					in, err := OpenInputs("s.yaml", r, 0)
					if err != nil {
						continue
					}
					if _, err := New(in); err != nil {
						continue
					}
					out = append(out, perturbationCase{fmt.Sprintf("%s/%s/tp%d/dp%d/ep%t/dcp%d/pcp%d/bs%d", m, c, tp, dp, ep, lay.dcp, lay.pcp, st.BlockSize), in})
					built = true
					break
				}
				_ = built
			}
		}
	}
	return out
}

// perturbationLayouts are the parallel layouts the perturbation prices: tensor parallel alone,
// expert parallel over data-parallel ranks (where all-to-all, reduce-scatter and the expert
// all-gathers run), and the two context-parallel forms.
var perturbationLayouts = []layoutCase{
	{tp: 2, dp: 1}, {tp: 4, dp: 1}, {tp: 8, dp: 1},
	{tp: 2, dp: 4, ep: true}, {tp: 1, dp: 8, ep: true}, {tp: 4, dp: 2, ep: true},
	{tp: 8, dp: 1, dcp: 2}, {tp: 8, dp: 1, dcp: 4}, {tp: 8, dp: 1, pcp: 2},
}

// perturbationSettings are the engine settings tried for a layout, in order; the first the cost
// model builds is the one priced. The backends a layout can route through come first, so an
// expert-parallel layout is priced on the all-to-all that uses the registry's all-to-all
// entries before the dense dispatch, which does not.
func perturbationSettings(lay layoutCase) []enginesettings.Settings {
	var out []enginesettings.Settings
	blocks := []enginesettings.Settings{
		{BlockSize: 16}, {BlockSize: 64}, {BlockSize: 256},
		{BlockSize: 64, CacheDType: "fp8_ds_mla"}, {BlockSize: 256, CacheDType: "fp8_ds_mla"},
	}
	if lay.ep {
		for _, backend := range []string{"deepep_high_throughput", "deepep_low_latency", "mori"} {
			for _, b := range blocks {
				b.All2AllBackend = backend
				out = append(out, b)
			}
		}
	}
	return append(out, blocks...)
}

// ctxPar renders the context-parallel widths of a layout, or nothing when both are off.
func ctxPar(l layoutCase) string {
	var b strings.Builder
	if l.dcp > 1 {
		fmt.Fprintf(&b, ", dcp: %d", l.dcp)
	}
	if l.pcp > 1 {
		fmt.Fprintf(&b, ", pcp: %d", l.pcp)
	}
	return b.String()
}

// layoutCase is a tensor-parallel width, a data-parallel width and whether expert parallelism is on.
type layoutCase struct {
	tp, dp int
	ep     bool
	dcp    int // decode-context-parallel width; 0 or 1 means off
	pcp    int // prefill-context-parallel width; 0 or 1 means off
}

// SUB-20's second check: changing a node's label changes no price. Every node of every
// catalog graph that carries a Role has it replaced by a string that matches nothing, and the
// model is priced again. A price that moves is a decision that depends on the label.
//
// At this release it does move for the models that have a block-index scorer or a shared
// expert, because the cost model branches on those two labels (contracts/testdata/
// role_branches.txt). The test lists them: it fails on a model not listed and on a listed model
// whose price no longer moves, so typed roles (blis-schemas#67) must delete lines.
func TestChangingALabelChangesNoPriceOrIsListed(t *testing.T) {
	if testing.Short() {
		t.Skip("prices every catalog model; CI runs it without -short")
	}
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	cases := perturbationScenarios(t, d)
	seen := map[string]bool{}
	var movers []string
	for _, c := range cases {
		model := strings.SplitN(c.label, "/", 2)[0]
		if seen[model] {
			continue
		}
		seen[model] = true
		base, err := New(c.in)
		if err != nil {
			t.Fatalf("%s: %v", c.label, err)
		}
		want := pricesOf(base)
		g := relabelled(t, *c.in.Model)
		in := c.in
		in.Model = &g
		k, err := New(in)
		if err != nil {
			t.Fatalf("%s with every role renamed: %v", c.label, err)
		}
		if pricesOf(k) != want {
			movers = append(movers, model)
		}
	}
	sort.Strings(movers)
	got := strings.Join(movers, "\n") + "\n"
	path := filepath.Join("testdata", "label_dependent_models.txt")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if os.Getenv("UPDATE_COMPLETENESS") != "" {
		var head []string
		for _, l := range strings.Split(string(raw), "\n") {
			if strings.HasPrefix(l, "#") {
				head = append(head, l)
			}
		}
		if err := os.WriteFile(path, []byte(strings.Join(head, "\n")+"\n"+got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	if want := withoutCommentLines(string(raw)); want != got {
		t.Errorf("the models whose price depends on a node label differ from %s.\nwant:\n%s\ngot:\n%s", path, want, got)
	}
}

// relabelled returns a copy of the graph with every node Role replaced by a string no code
// compares with, and every reference to a role (a node's Aliases, an architecture override's
// Role) rewritten to match, so the document stays valid and only the label changes. It walks
// the whole graph by reflection, so the speculator's own stacks, the embedding, the head and
// the architecture overrides are renamed with the layer kinds. The empty role stays empty: its
// absence is itself what several sites read.
func relabelled(t *testing.T, g model.Graph) model.Graph {
	t.Helper()
	b, err := yaml.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	var out model.Graph
	if err := yaml.Unmarshal(b, &out); err != nil {
		t.Fatalf("copying the graph: %v", err)
	}
	rename := map[string]string{}
	visit(reflect.ValueOf(&out), func(f reflect.StructField, v reflect.Value) {
		if f.Name == "Role" && v.Kind() == reflect.String && v.String() != "" {
			if _, ok := rename[v.String()]; !ok {
				rename[v.String()] = fmt.Sprintf("relabelled_%d", len(rename)+1)
			}
		}
	})
	visit(reflect.ValueOf(&out), func(f reflect.StructField, v reflect.Value) {
		switch {
		case f.Name == "Role" && v.Kind() == reflect.String && v.String() != "":
			v.SetString(rename[v.String()])
		case f.Name == "Aliases" && v.Kind() == reflect.String && v.String() != "":
			if scope, role, ok := strings.Cut(v.String(), "."); ok {
				if nr, found := rename[role]; found {
					v.SetString(scope + "." + nr)
				}
			}
		}
	})
	return out
}

// visit calls fn for every settable struct field reachable from v.
func visit(v reflect.Value, fn func(reflect.StructField, reflect.Value)) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			visit(v.Elem(), fn)
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			visit(v.Index(i), fn)
		}
	case reflect.Map:
		for _, k := range v.MapKeys() {
			visit(v.MapIndex(k), fn)
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			f := v.Type().Field(i)
			if !f.IsExported() {
				continue
			}
			fn(f, v.Field(i))
			visit(v.Field(i), fn)
		}
	}
}

// withoutCommentLines drops the comment lines of a testdata list.
func withoutCommentLines(s string) string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if !strings.HasPrefix(l, "#") {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}
