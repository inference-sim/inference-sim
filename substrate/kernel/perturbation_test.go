package latencykernel

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
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
	// A transfer to and from each tier, at a small and a large size and at two queue depths.
	tiers := make([]string, 0, len(k.tiers))
	for name := range k.tiers {
		tiers = append(tiers, name)
	}
	sort.Strings(tiers)
	for _, tier := range tiers {
		for _, dir := range []kernel.Direction{kernel.DirectionFromTier, kernel.DirectionToTier} {
			for _, bytes := range []int64{4096, 1 << 30} {
				for _, qd := range []int{1, 4} {
					fmt.Fprintf(&b, "tier %s %s %d %d %d|", tier, dir, bytes, qd, k.TierTime(tier, dir, bytes, qd))
				}
			}
		}
	}
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
func perturbationScenarios(t *testing.T, d *load.Data, only ...string) []perturbationCase {
	t.Helper()
	// Every deployment offloads to every catalog tier, so each tier's three coefficients are
	// read by a lookup and perturbed with the rest (pricesOf prices a transfer to each).
	offload := "offload:\n  tiers:\n"
	for _, dev := range d.StorageDevices() {
		offload += fmt.Sprintf("    - {tier: %s, bytes: 1073741824}\n", dev.Name)
	}
	sets := "[cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric, cost-model-offload-tiers]"
	var out []perturbationCase
	for _, m := range d.Models() {
		if len(only) > 0 && !slices.Contains(only, m) {
			continue
		}
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
					body := fmt.Sprintf("kind: Scenario\nname: p\nmodel: %s\ncoefficients: %s\nengine_version: \"0.32.0\"\ncluster:\n  hardware: %s\n  fabric: ib-400g\n  nodes: %d\n%s---\nkind: Deployment\nname: d\npools:\n  - role: colocated\n    nodes: %d\n    parallel: {tp: %d, pp: 1, dp: %d, enable_expert_parallel: %t%s}\n%s", m, sets, c, nodes, packaging, nodes, tp, dp, ep, ctxPar(lay), offload)
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
					label := fmt.Sprintf("%s/%s/tp%d/dp%d/ep%t/dcp%d/pcp%d/bs%d", m, c, tp, dp, ep, lay.dcp, lay.pcp, st.BlockSize)
					out = append(out, perturbationCase{label, in})
					// The same scenario with the trtllm-gen prefill lane priced, where the chip
					// carries it: the lane's bf16 pair is read at an auto cache and the fp8 pair
					// only at an fp8 cache, so each is perturbed here. The registry's lane states no
					// kernel implementation and is not priced (VAL-8), so these cases key it to a
					// fixture implementation that the pool reports running.
					laneIn := keyedLane(in, laneBF16)
					if _, err := New(laneIn); err == nil {
						out = append(out, perturbationCase{label + "/trtllm", laneIn})
					} // And at an fp8 cache, which reads the lane's fp8 pair.
					f8 := st.Clone()
					f8.CacheDType = "fp8"
					f8In := keyedLane(in, laneFP8)
					f8In.Engine = &f8
					if _, err := New(f8In); err == nil {
						out = append(out, perturbationCase{label + "/trtllm-fp8", f8In})
					}
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

// family is a coefficient name without the rank width and chip that name a collective's entry:
// the quantity the registry measures, which is what a perturbation asks about.
func family(name string) string {
	return regexp.MustCompile(`_(\d+rank)`).ReplaceAllString(name, "_<R>rank")
}

// perturbed returns the scenario's coefficient sets with every entry of the given name scaled
// by 0.8, leaving the others untouched. A smaller value keeps efficiencies and fractions valid.
func perturbed(sets []*coefficient.Set, name string) []*coefficient.Set {
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if c.Coefficients[j].Name == name {
				c.Coefficients[j].Value *= 0.8
				if iv := c.Coefficients[j].CI95; iv != nil {
					// The interval scales with the value, or the entry would lie outside its own.
					c.Coefficients[j].CI95 = &coefficient.Interval{Low: iv.Low * 0.8, High: iv.High * 0.8}
				}
			}
		}
		out[i] = &c
	}
	return out
}

// SUB-11: every coefficient family changes some price when it is changed, or is listed as not
// yet priced. Each scenario is built, then each coefficient its lookups returned is scaled by
// 0.8 and the scenario priced again. A family is priced when any scenario's prices move. A
// family no scenario reads, or one whose every read left every price unchanged, is a gap in
// the cost model or in these scenarios, and is listed in testdata/unpriced_families.txt. The
// list may only shrink: the test fails on a family not listed and on a listed family that is
// now priced.
func TestEveryCoefficientFamilyChangesSomePriceOrIsListed(t *testing.T) {
	if testing.Short() {
		t.Skip("prices about 1,000 scenarios and takes about a minute; CI runs it without -short")
	}
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	scenarios := perturbationScenarios(t, d)
	if len(scenarios) < 400 {
		t.Fatalf("only %d scenarios build; the perturbation is not reaching the kernel", len(scenarios))
	}
	priced := map[string]bool{}
	read := map[string]bool{}
	for _, s := range scenarios {
		base, err := New(s.in)
		if err != nil {
			t.Fatalf("%s: %v", s.label, err)
		}
		want := pricesOf(base)
		for _, sel := range base.selected {
			name := sel.Entry.Name
			fam := family(name)
			read[fam] = true
			if priced[fam] || sel.Entry.Value == 0 {
				continue
			}
			in := s.in
			in.Coefficients = perturbed(s.in.Coefficients, name)
			k, err := New(in)
			if err != nil {
				t.Fatalf("%s: scaling %s by 0.8 makes the kernel refuse: %v", s.label, name, err)
			}
			if pricesOf(k) != want {
				priced[fam] = true
			}
		}
	}
	all := map[string]bool{}
	for _, set := range []string{"cost-model-primitives", "cost-model-collectives", "cost-model-host-overheads",
		"cost-model-attention", "cost-model-recurrent", "cost-model-memory", "cost-model-fabric"} {
		cs, err := d.CoefficientSet(set)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range cs.Coefficients {
			if e.Value != 0 {
				all[family(e.Name)] = true
			}
		}
	}
	var unpriced []string
	for fam := range all {
		if !priced[fam] {
			why := "read but changes no price"
			if !read[fam] {
				why = "never read"
			}
			unpriced = append(unpriced, fam+" # "+why)
		}
	}
	sort.Strings(unpriced)
	got := strings.Join(unpriced, "\n") + "\n"
	path := filepath.Join("testdata", "unpriced_families.txt")
	if os.Getenv("UPDATE_COMPLETENESS") != "" {
		old, _ := os.ReadFile(path)
		var head []string
		for _, l := range strings.Split(string(old), "\n") {
			if strings.HasPrefix(l, "#") {
				head = append(head, l)
			}
		}
		out := got
		if len(head) > 0 {
			out = strings.Join(head, "\n") + "\n" + got
		}
		if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if want := stripComments(string(raw)); want != got {
		t.Errorf("the families that change no price differ from testdata/unpriced_families.txt.\nwant:\n%s\ngot:\n%s", want, got)
	}
	t.Logf("%d scenarios, %d families, %d unpriced", len(scenarios), len(all), len(unpriced))
}

// SUB-20's second check: changing a node's label changes no price. Every node of every
// catalog graph that carries a Role has it replaced by a string that matches nothing, and the
// model is priced again. A price that moves is a decision that depends on the label.
//
// No catalog model's price moves: the cost model reads holds_engine_kv and shared, which the
// deriver sets, and no label. testdata/label_dependent_models.txt is empty; the test still
// fails on a model that moves and is not listed, so a new decision on a label cannot return.
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
	got := ""
	if len(movers) > 0 {
		got = strings.Join(movers, "\n") + "\n"
	}
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
	if want := stripComments(string(raw)); want != got {
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
