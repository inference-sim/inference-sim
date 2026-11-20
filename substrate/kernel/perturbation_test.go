package latencykernel

import (
	"fmt"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
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
	fmt.Fprintf(&b, "mem %d %d %d %d %d %d|seq %d %d|", f.Weights, f.ActivationPeak, f.CUDAGraph, f.EPLBRedundant, f.CommBuffers,
		f.NonTorchInBudget, k.SequenceFixedBytes(), k.SequenceVariableBytes(4096))
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
	// Each (model, chip) pair is built on its own, so the pairs are built on every CPU; the
	// results are joined in the order of the pairs, the order the sequential loop produced.
	pairs := perturbationPairs(d, only)
	built := make([][]perturbationCase, len(pairs))
	forEachPerturbationPair(t, d, pairs, runtime.GOMAXPROCS(0), func(i int, cases []perturbationCase) { built[i] = cases })
	var out []perturbationCase
	for i := range pairs {
		out = append(out, built[i]...)
	}
	return out
}

// perturbationTexts are the offload block and the coefficient-set list every perturbation scenario states.
func perturbationTexts(d *load.Data) (offload, sets string) {
	// Every deployment offloads to every catalog tier, so each tier's three coefficients are
	// read by a lookup and perturbed with the rest (pricesOf prices a transfer to each).
	offload = "offload:\n  tiers:\n"
	for _, dev := range d.StorageDevices() {
		offload += fmt.Sprintf("    - {tier: %s, bytes: 1073741824}\n", dev.Name)
	}
	sets = "[cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric, cost-model-offload-tiers]"
	return offload, sets
}

// perturbationPair is a catalog model and a catalog chip.
type perturbationPair struct{ model, chip string }

// perturbationPairs lists the (model, chip) pairs the perturbation scenarios are built from.
func perturbationPairs(d *load.Data, only []string) []perturbationPair {
	var pairs []perturbationPair
	for _, m := range d.Models() {
		if len(only) > 0 && !slices.Contains(only, m) {
			continue
		}
		for _, c := range d.Chips() {
			pairs = append(pairs, perturbationPair{m, c})
		}
	}
	return pairs
}

// forEachPerturbationPair builds the scenarios of each pair on the given number of workers and
// hands them to use, which runs on the worker that built them: a caller that prices them there
// holds a pair's scenarios at a time per worker, not every scenario of the catalog at once. A pair
// that cannot be built fails the test (the first in the list of pairs reported).
func forEachPerturbationPair(t *testing.T, d *load.Data, pairs []perturbationPair, workers int, use func(i int, cases []perturbationCase)) {
	t.Helper()
	offload, sets := perturbationTexts(d)
	errs := make([]error, len(pairs))
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				var cases []perturbationCase
				cases, errs[i] = perturbationScenariosFor(t, d, pairs[i].model, pairs[i].chip, offload, sets)
				if errs[i] == nil {
					use(i, cases)
				}
			}
		}()
	}
	for i := range pairs {
		next <- i
	}
	close(next)
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

// customLaneModel picks about one model in eight, by its name, to also price the custom all-reduce
// lane: the lane's entries are named by part and width, so a few models per part read them all, and a
// variant of every scenario would double the memory and time of this test on a hosted runner.
func customLaneModel(m string) bool {
	h := 0
	for _, r := range m {
		h = h*31 + int(r)
	}
	return h%8 == 0
}

// perturbationScenariosFor builds the scenarios of one model on one chip.
func perturbationScenariosFor(t *testing.T, d *load.Data, m, c, offload, sets string) ([]perturbationCase, error) {
	var out []perturbationCase
	ch, _ := d.Chip(c)
	// Layouts: tensor parallel alone at 2, 4 and 8; and expert parallel over data parallel
	// ranks, which is where all-to-all, reduce-scatter and the expert all-gathers run.
	for _, lay := range perturbationLayouts {
		tp, dp, ep := lay.tp, lay.dp, lay.ep
		built := false
		for _, st := range perturbationSettings(lay) {
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
				return nil, err
			}
			r := Repos{Scenarios: dir, Data: d, Rules: func(string) rules.EngineRules { return windowedRules{} },
				DraftModel: noDrafter, Lower: lowerTo(8192, st.block, 0, st.edit)}
			in, err := OpenInputs("s.yaml", r, 0)
			if err != nil {
				continue
			}
			if _, err := New(in); err != nil {
				continue
			}
			label := fmt.Sprintf("%s/%s/tp%d/dp%d/ep%t/dcp%d/pcp%d/bs%d", m, c, tp, dp, ep, lay.dcp, lay.pcp, st.block)
			out = append(out, perturbationCase{label, in})
			// The lowered plans these scenarios state no all-reduce kernel, which prices the NCCL
			// lane; the same scenario with the engine's custom kernel prices the other lane.
			if tp > 1 && dp == 1 && !ep && lay.dcp == 0 && lay.pcp == 0 && customLaneModel(m) {
				cu := *in.Lowered
				cu.Reported.SMAllReduce, cu.Plan.Collectives.AllReduce = true, plan.AllReduceCustom
				cuIn := in
				cuIn.Lowered = &cu
				if _, err := New(cuIn); err == nil {
					out = append(out, perturbationCase{label + "/custom-allreduce", cuIn})
				}
			}
			// The same scenario with the trtllm-gen prefill lane priced, where the chip
			// carries it: the lane's bf16 pair is read at an auto cache and the fp8 pair
			// only at an fp8 cache, so each is perturbed here. The registry's lane states no
			// kernel implementation and is not priced (VAL-8), so these cases key it to a
			// fixture implementation that the pool reports running.
			laneIn := keyedLane(in, laneBF16)
			if _, err := New(laneIn); err == nil {
				out = append(out, perturbationCase{label + "/trtllm", laneIn})
			} // And at an fp8 cache, which reads the lane's fp8 pair.
			f8 := *in.Lowered
			f8.Plan.Cache.Record, f8.Plan.Cache.ElementBytes, f8.Plan.Cache.LatentCellBytes = "fp8", 1, 0
			f8In := keyedLane(in, laneFP8)
			f8In.Lowered = &f8
			if _, err := New(f8In); err == nil {
				out = append(out, perturbationCase{label + "/trtllm-fp8", f8In})
			}
			// The GLM-5.3 family's sparse-MLA decode entries are keyed to the FlashInfer
			// decode build the engine pin runs; this case reports it running, so they are
			// read and perturbed (keyedSparseDecode).
			if sdIn, ok := keyedSparseDecode(in); ok {
				if _, err := New(sdIn); err == nil {
					out = append(out, perturbationCase{label + "/sparse-decode", sdIn})
				}
			}
			// And a DSA model at a packed fp8_ds_mla cache (dsa_prefill_fp8_test.go).
			if pk, ok := packedSparseVariant(in); ok {
				out = append(out, perturbationCase{label + "/packed", pk})
			}
			// The same scenario with each grouped-expert kernel the registry measures on
			// Hopper stated (plan.MoEPlan.Experts), on a part and model where the plan
			// reaches it: block-FP8 routed experts on SM90.
			if (c == "h100" || c == "h200") && hasGroupedExperts(in) {
				for _, name := range []string{"gemm_floor_fp8_grouped_triton", "gemm_floor_fp8_grouped_flashinfer_cutlass"} {
					v := pinnedBuild(t, d, name, c)
					ex := *in.Lowered
					ex.Plan.MoE.Experts = v
					exIn := in
					exIn.Lowered = &ex
					if _, err := New(exIn); err == nil {
						out = append(out, perturbationCase{label + "/" + v.Library, exIn})
					}
				}
			}
			built = true
			break
		}
		_ = built
	}
	return out, nil
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
func perturbationSettings(lay layoutCase) []perturbationSetting {
	var out []perturbationSetting
	blocks := []perturbationSetting{{block: 16}, {block: 64}, {block: 256}, {block: 64, packed: true},
		{block: 256, packed: true}}
	if lay.ep {
		for _, dispatch := range []plan.KernelImpl{plan.DispatchDeepEPHighThroughput,
			plan.DispatchDeepEPLowLatency, {Library: "mori"}} {
			for _, b := range blocks {
				b.dispatch = dispatch
				out = append(out, b)
			}
		}
	}
	return append(out, blocks...)
}

// perturbationSetting is one plan a perturbation tries: a page size, a cache stored as an fp8
// record (packed latent cells on a latent layer), and an MoE dispatch kernel.
type perturbationSetting struct {
	block    int
	packed   bool
	dispatch plan.KernelImpl
}

// edit states the setting in a plan.
func (s perturbationSetting) edit(l *plan.Lowered) {
	if s.packed {
		l.Plan.Cache.Record, l.Plan.Cache.ElementBytes, l.Plan.Cache.LatentCellBytes = "fp8_ds_mla", 1, 656
	}
	l.Plan.MoE.Dispatch = s.dispatch
}

// hasGroupedExperts reports whether the scenario's model routes tokens through grouped experts.
func hasGroupedExperts(in Inputs) bool {
	for _, lk := range in.Model.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Op == model.OpGroupedGEMM && !n.IsSharedExpert() {
				return true
			}
		}
	}
	return false
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
	return rankWidth.ReplaceAllString(name, "_<R>rank")
}

// rankWidth is the rank width inside a collective's coefficient name, compiled once: family runs
// for every entry every scenario's lookups return.
var rankWidth = regexp.MustCompile(`_(\d+rank)`)

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
	// A family is priced when any scenario's prices move and read when any scenario reads it, so
	// the answer does not depend on the order the scenarios are priced in. Each worker builds the
	// scenarios of one (model, chip) pair and prices them before it builds the next, sharing the
	// two sets, so only a pair's scenarios per worker are held at once; a family already seen to
	// move a price is not tried again, as in the sequential loop. A scenario that cannot be priced
	// fails the test, the first pair in the list reported. At most two workers: each holds the
	// kernels it builds and the collector's headroom over them, about 3 GB, and the runner that
	// executes this test has 16 GB (the test was killed there with every CPU working).
	var pricedSet, readSet sync.Map
	var built atomic.Int64
	pairs := perturbationPairs(d, nil)
	errAt := make([]error, len(pairs))
	forEachPerturbationPair(t, d, pairs, min(runtime.GOMAXPROCS(0), 2), func(pi int, cases []perturbationCase) {
		built.Add(int64(len(cases)))
		for _, s := range cases {
			base, err := New(s.in)
			if err != nil {
				errAt[pi] = fmt.Errorf("%s: %v", s.label, err)
				return
			}
			want := pricesOf(base)
			// A name is perturbed once per scenario: lookups return the same entry for many
			// layers, and scaling the same entry again prices the same scenario again.
			tried := map[string]bool{}
			for _, sel := range base.selected {
				name := sel.Entry.Name
				fam := family(name)
				readSet.Store(fam, true)
				if _, done := pricedSet.Load(fam); done || sel.Entry.Value == 0 || tried[name] {
					continue
				}
				tried[name] = true
				in := s.in
				in.Coefficients = perturbed(s.in.Coefficients, name)
				k, err := New(in)
				if err != nil {
					errAt[pi] = fmt.Errorf("%s: scaling %s by 0.8 makes the kernel refuse: %v", s.label, name, err)
					return
				}
				if pricesOf(k) != want {
					pricedSet.Store(fam, true)
				}
			}
		}
	})
	if built.Load() < 400 {
		t.Fatalf("only %d scenarios build; the perturbation is not reaching the kernel", built.Load())
	}
	for _, err := range errAt {
		if err != nil {
			t.Fatal(err)
		}
	}
	priced := map[string]bool{}
	read := map[string]bool{}
	pricedSet.Range(func(k, _ any) bool { priced[k.(string)] = true; return true })
	readSet.Range(func(k, _ any) bool { read[k.(string)] = true; return true })
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
	t.Logf("%d scenarios, %d families, %d unpriced", built.Load(), len(all), len(unpriced))
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
	// One scenario per model: the first the catalog's chips give it, in the order of d.Chips(). Only that
	// scenario is kept; building every scenario of the catalog to pick these held about 9 GB.
	offload, sets := perturbationTexts(d)
	models := d.Models()
	first := make([]perturbationCase, len(models))
	has := make([]bool, len(models))
	var buildErr = make([]error, len(models))
	{
		var bw sync.WaitGroup
		nb := make(chan int)
		for w := 0; w < min(runtime.GOMAXPROCS(0), 2); w++ {
			bw.Add(1)
			go func() {
				defer bw.Done()
				for i := range nb {
					for _, c := range d.Chips() {
						cases, err := perturbationScenariosFor(t, d, models[i], c, offload, sets)
						if err != nil {
							buildErr[i] = err
							break
						}
						if len(cases) > 0 {
							first[i], has[i] = cases[0], true
							break
						}
					}
				}
			}()
		}
		for i := range models {
			nb <- i
		}
		close(nb)
		bw.Wait()
	}
	var keep []perturbationCase
	for i := range models {
		if buildErr[i] != nil {
			t.Fatal(buildErr[i])
		}
		if has[i] {
			keep = append(keep, first[i])
		}
	}
	first = keep
	// Each is priced twice (as built and with every role renamed), on two workers; the movers
	// are joined in the order of the models, then sorted.
	moved := make([]bool, len(first))
	failed := make([]string, len(first))
	var wg sync.WaitGroup
	next := make(chan int)
	// Two workers: each prices a whole model's graph twice, and the runners that execute this
	// test have little memory to give several at once.
	for w := 0; w < min(runtime.GOMAXPROCS(0), 2); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				c := first[i]
				base, err := New(c.in)
				if err != nil {
					failed[i] = fmt.Sprintf("%s: %v", c.label, err)
					continue
				}
				want := pricesOf(base)
				g, err := relabelledGraph(*c.in.Model)
				if err != nil {
					failed[i] = fmt.Sprintf("%s: %v", c.label, err)
					continue
				}
				in := c.in
				in.Model = &g
				k, err := New(in)
				if err != nil {
					failed[i] = fmt.Sprintf("%s with every role renamed: %v", c.label, err)
					continue
				}
				moved[i] = pricesOf(k) != want
			}
		}()
	}
	for i := range first {
		next <- i
	}
	close(next)
	wg.Wait()
	var movers []string
	for i, c := range first {
		if failed[i] != "" {
			t.Fatal(failed[i])
		}
		if moved[i] {
			movers = append(movers, strings.SplitN(c.label, "/", 2)[0])
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

// relabelledGraph returns a copy of the graph with every node Role replaced by a string no code
// compares with, and every reference to a role (a node's Aliases, an architecture override's
// Role) rewritten to match, so the document stays valid and only the label changes. It walks
// the whole graph by reflection, so the speculator's own stacks, the embedding, the head and
// the architecture overrides are renamed with the layer kinds. The empty role stays empty: its
// absence is itself what several sites read.
func relabelledGraph(g model.Graph) (model.Graph, error) {
	b, err := yaml.Marshal(g)
	if err != nil {
		return model.Graph{}, err
	}
	var out model.Graph
	if err := yaml.Unmarshal(b, &out); err != nil {
		return model.Graph{}, fmt.Errorf("copying the graph: %v", err)
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
	return out, nil
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
