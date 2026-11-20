package cmd

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Behavioral contract for the cluster matrix (M3.2 Table W row 28, requirements section 5):
// every model in the catalog, on each of six chips, over each chip's two fabrics, in aggregated
// and in P/D serving, for each engine, either simulates or is refused with the upstream's reason.
//
//	M1 - a cell recorded as running completes every request it injects, and every pool's price
//	     count names a method from the registry's vocabulary and sums to the pool's total.
//	M2 - a cell recorded as refused is refused again, with the reason (and the issue the reason
//	     carries) recorded for it; a refusal that is not a capacity limit is a defect and has an
//	     issue.
//	M3 - the fabric a cell is deployed on does not change whether it runs: a cell over RoCE is
//	     refused exactly when its InfiniBand twin is.
//	M4 - docs/support-matrix.tsv names every catalog model once per engine, chip, fabric and
//	     serving mode, and docs/support-matrix.md is what the generator renders from it.
//
// docs/support-matrix.tsv is the expected outcome of each cell and the published support matrix.
// `go test ./cmd -run ClusterMatrix -update-matrix` searches each model's smallest tensor-parallel
// size that fits, runs every cell, and rewrites both files; BLIS_MATRIX_FULL=1 runs every
// recorded cell instead of the per-pull-request subset.

var updateMatrix = flag.Bool("update-matrix", false, "rewrite docs/support-matrix.{tsv,md} from a search of every cell")

const (
	matrixTSV = "../docs/support-matrix.tsv"
	matrixMD  = "../docs/support-matrix.md"

	matrixRequests     = 4
	matrixPromptTokens = 256
	matrixOutputTokens = 16
	matrixWorkers      = 3
	matrixMaxModelLen  = 4096
	matrixSampleEvery  = 8
)

var (
	matrixTPs        = []int{1, 2, 4, 8, 16, 24, 32, 48, 64}
	matrixBlockSizes = []int{16, 64, 256}
	matrixMethods    = []string{"measured", "calibrated", "transferred", "literature", "vendor_spec", "roofline", "assumed", "not_charged"}
	matrixEngines    = []string{"vllm", "sglang"}
	matrixModes      = []string{"agg", "pd"}
)

type matrixChip struct {
	name string
	// gpusPerNode is the node size a scenario must state (8 for the HGX boards); a chip whose
	// catalog entry fixes the packaging (the NVL72 trays) states none.
	gpusPerNode int
	statesGPN   bool
	rate        string
}

var matrixChips = []matrixChip{
	{"h100", 8, true, "400g"},
	{"h200", 8, true, "400g"},
	{"b200", 8, true, "400g"},
	{"b300", 8, true, "800g"},
	{"gb200-nvl72", 4, false, "400g"},
	{"gb300", 4, false, "800g"},
}

func matrixChipOf(name string) matrixChip {
	for _, c := range matrixChips {
		if c.name == name {
			return c
		}
	}
	panic("unknown chip " + name)
}

// matrixCell is one row of the support matrix.
type matrixCell struct {
	Model, Engine, Chip, Fabric, Mode string
	TP, BlockSize                     int
	Outcome                           string // "runs" or "refused"
	Class, Issue                      string // of a refusal
	Methods                           string // "method=count;..." of the cell's prices, summed over its pools
	Detail                            string // the refusal's reason
}

func (c matrixCell) key() string {
	return strings.Join([]string{c.Model, c.Engine, c.Chip, c.Fabric, c.Mode}, "/")
}

func (c matrixCell) isRoCE() bool { return strings.HasPrefix(c.Fabric, "roce-") }

// matrixClass sorts a refusal's reason. step classes are limits of one tensor-parallel size or
// block size that the search climbs past; the rest end it. defect is whether BLIS refuses
// for lack of a modelled feature (an issue holds it) rather than because the model cannot be
// served by the cluster (requirements 5.2).
type matrixClass struct {
	id     string
	re     *regexp.Regexp
	step   string // "block" or "tp" for a search step
	defect bool
	// issue is the open issue that holds the defect; a reason that carries its own reference
	// ("(inference-sim/blis-sglang#49)") is read from the message instead.
	issue string
	why   string
}

var matrixClasses = []matrixClass{
	{id: "block_size_multiple", re: regexp.MustCompile(`block_size \d+ on a .*a stated size must be a multiple`), step: "block",
		why: "the stated block size is not a multiple of the attention kernel's block"},
	{id: "capacity", re: regexp.MustCompile(`does not fit [\d.]+ GiB devices|No available memory for the cache blocks|loaded weights leave no GPU memory for the KV cache|larger than the available KV c|per-rank fixed occupancy`), step: "tp",
		why: "the model does not fit the device at this tensor-parallel size"},
	{id: "kv_heads_not_divisible", re: regexp.MustCompile(`does not divide the \d+ KV heads of`), step: "tp",
		why: "the tensor-parallel size does not divide the KV head count of a projection that cannot replicate a head"},
	{id: "heads_not_divisible", re: regexp.MustCompile(`is not divisible by tensor_parallel_size`), step: "tp",
		why: "the tensor-parallel size does not divide the head count"},
	{id: "sglang_pd_not_modelled", re: regexp.MustCompile(`prefill/decode disaggregation is not modelled yet \(inference-sim/blis-sglang#\d+\)`), defect: true,
		why: "the SGLang engine does not model prefill/decode disaggregation"},
	{id: "sglang_pd_not_placeable", re: regexp.MustCompile(`the engine does not implement engine\.Placeable`), defect: true, issue: "inference-sim/blis-sglang#64",
		why: "the SGLang engine does not model prefill/decode disaggregation, so a pool of several nodes is refused at placement before a request is"},
	{id: "sglang_not_lowered", re: regexp.MustCompile(`\(inference-sim/blis-sglang#\d+\)`), defect: true,
		why: "the SGLang engine does not lower this architecture, chip or setting"},
	{id: "recurrent_sparse_mla", re: regexp.MustCompile(`sparse_mla attention beside recurrent layers is not mapped`), defect: true,
		issue: "inference-sim/blis-vllm#308", why: "sparse MLA attention beside recurrent layers is not mapped"},
	{id: "recurrent_head_size_256", re: regexp.MustCompile(`an attention head size 256 beside recurrent layers`), defect: true,
		issue: "inference-sim/blis-vllm#309", why: "a 256-wide attention head beside recurrent layers: the kernel block depends on the FlashAttention version"},
	{id: "upstream_no_kv_connector", re: regexp.MustCompile(`supports_kv_connector False`),
		why: "vLLM's own attention backend for this model supports no KV connector, so it cannot serve P/D"},
	{id: "kv_layers_remainder", re: regexp.MustCompile(`kv_layers: the memory model's .* are not a whole number of bytes`), defect: true,
		issue: "inference-sim/blis-vllm#310", why: "the KV layer sizing leaves a remainder that is not a whole number of bytes for the other layers"},
}

var matrixIssueRef = regexp.MustCompile(`inference-sim/[a-z-]+#\d+`)

// classify returns the class of a refusal's reason and the issue that holds it.
func classify(reason string) (matrixClass, string, bool) {
	for _, c := range matrixClasses {
		if !c.re.MatchString(reason) {
			continue
		}
		issue := c.issue
		if c.defect && issue == "" {
			issue = matrixIssueRef.FindString(reason)
		}
		return c, issue, true
	}
	return matrixClass{}, "", false
}

// matrixResult is the part of a Result the matrix reads.
type matrixResult struct {
	Requests struct {
		Generated int `json:"generated"`
		Completed int `json:"completed"`
		Dropped   struct {
			Unservable int `json:"unservable"`
		} `json:"dropped"`
	} `json:"requests"`
	Reports struct {
		Kernel map[string]struct {
			MethodCounts map[string]int `json:"method_counts"`
			Total        int            `json:"total"`
		} `json:"kernel"`
	} `json:"reports"`
}

type matrixOutcome struct {
	ok      bool
	reason  string
	methods map[string]int
	err     string // a malformed success
}

type matrixRunner struct {
	blis string
	dir  string
}

func newMatrixRunner(t *testing.T) *matrixRunner {
	t.Helper()
	dir := t.TempDir()
	pick, err := os.ReadFile("../testdata/scenarios/router/pd-always.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "router"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "router", "pd-always.yaml"), pick, 0o644); err != nil {
		t.Fatal(err)
	}
	return &matrixRunner{blis: builtBlis(t), dir: dir}
}

// matrixScenario is the Scenario + Deployment document of one cell. The engine settings are the
// ones each engine requires and no more; the search records the tensor-parallel size and block
// size that make the cell run.
func matrixScenario(c matrixCell) (name, doc string) {
	chip := matrixChipOf(c.Chip)
	name = strings.NewReplacer(".", "_", "/", "-").Replace(fmt.Sprintf("%s-%s-%s-%s-%s-tp%d-b%d", c.Model, c.Engine, c.Chip, c.Fabric, c.Mode, c.TP, c.BlockSize))
	perPool := (c.TP + chip.gpusPerNode - 1) / chip.gpusPerNode
	var engine string
	switch c.Engine {
	case "vllm":
		engine = fmt.Sprintf("    engine:\n      block_size: %d\n      max_num_batched_tokens: 8192\n      max_num_seqs: 256\n      max_model_len: %d\n      gpu_memory_utilization: 0.9\n      compilation_config: {cudagraph_mode: PIECEWISE}\n", c.BlockSize, matrixMaxModelLen)
	default:
		engine = fmt.Sprintf("    engine:\n      mem_fraction_static: 0.85\n      context_length: %d\n", matrixMaxModelLen)
	}
	parallel := fmt.Sprintf("    parallel: {tp: %d, pp: 1, dp: 1, enable_expert_parallel: false}\n", c.TP)
	var pools, tail string
	nodes := perPool
	if c.Mode == "agg" {
		pools = fmt.Sprintf("  - role: colocated\n    nodes: %d\n%s%s", perPool, parallel, engine)
	} else {
		// An 8-GPU chip's two pools take whole nodes of their own, so the handoff crosses the
		// fabric. An NVL72 rack holds 18 trays, so the prefill pool fills the rack as far as its
		// instances divide it and the decode pool starts beyond the rack (or across its end),
		// where the handoff leaves the NVLink domain.
		prefill := perPool
		if !chip.statesGPN {
			prefill = (18 / perPool) * perPool
		}
		nodes = prefill + perPool
		pools = fmt.Sprintf("  - role: prefill\n    nodes: %d\n%s%s  - role: decode\n    nodes: %d\n%s%s", prefill, parallel, engine, perPool, parallel, engine)
		tail = "\npd_transfer:\n  connector: NixlConnector\n\nrouter:\n  endpoint_picker_config: router/pd-always.yaml\n"
	}
	var gpn string
	if chip.statesGPN {
		gpn = "  gpus_per_node: 8\n"
	}
	doc = fmt.Sprintf(`kind: Scenario
name: %s
engine: {kind: %s}
model: %s
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
cluster:
  hardware: %s
  fabric: %s
  nodes: %d
%s---
kind: Deployment
name: %s
pools:
%s%s`, name, c.Engine, c.Model, c.Chip, c.Fabric, nodes, gpn, name, pools, tail)
	return name, doc
}

var matrixFatal = regexp.MustCompile(`level=(fatal|error) .*msg=(.*)$`)

// matrixEngineRefused is an engine's refusal of a request, which a run reports as dropped
// requests rather than a startup failure (SGLang prefill/decode disaggregation).
var matrixEngineRefused = regexp.MustCompile(`(?m)the engine refused it: (.*?)(?:: request \S+)?"?$`)
var matrixScenarioPrefix = regexp.MustCompile(`scenario \\?"[^ ]*?\\?": |[a-z0-9._-]+-tp\d+-b\d+\.yaml: `)

// run simulates a cell. A failure to start the process is a test failure; the process refusing
// the cell is the outcome.
func (r *matrixRunner) run(t *testing.T, c matrixCell) matrixOutcome {
	name, doc := matrixScenario(c)
	file := name + ".yaml"
	if err := os.WriteFile(filepath.Join(r.dir, file), []byte(doc), 0o644); err != nil {
		return matrixOutcome{reason: "writing the scenario: " + err.Error()}
	}
	cmd := exec.Command(r.blis, "run", "--scenarios", r.dir, "--scenario", file,
		"--rate", "4", "--num-requests", fmt.Sprint(matrixRequests),
		"--prompt-tokens", fmt.Sprint(matrixPromptTokens), "--output-tokens", fmt.Sprint(matrixOutputTokens))
	var out, errBuf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errBuf
	err := cmd.Run()
	_ = os.Remove(filepath.Join(r.dir, file))
	if err != nil {
		if _, isExit := err.(*exec.ExitError); !isExit {
			return matrixOutcome{reason: "starting blis: " + err.Error()}
		}
		return matrixOutcome{reason: refusalReason(errBuf.String())}
	}
	var res matrixResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		return matrixOutcome{ok: true, err: "stdout is not a Result: " + err.Error()}
	}
	if res.Requests.Completed == 0 && res.Requests.Dropped.Unservable == matrixRequests {
		if m := matrixEngineRefused.FindStringSubmatch(errBuf.String()); m != nil {
			return matrixOutcome{reason: m[1]}
		}
	}
	o := matrixOutcome{ok: true, methods: map[string]int{}}
	if res.Requests.Generated != matrixRequests || res.Requests.Completed != matrixRequests {
		o.err = fmt.Sprintf("injected %d and completed %d of %d requests", res.Requests.Generated, res.Requests.Completed, matrixRequests)
	}
	if len(res.Reports.Kernel) == 0 {
		o.err = "the Result has no kernel report"
	}
	for pool, k := range res.Reports.Kernel {
		sum := 0
		for method, n := range k.MethodCounts {
			if !contains(matrixMethods, method) {
				o.err = fmt.Sprintf("pool %s: price method %q is not in the registry's vocabulary", pool, method)
			}
			sum += n
			o.methods[method] += n
		}
		if k.Total == 0 || sum != k.Total {
			o.err = fmt.Sprintf("pool %s: the method counts sum to %d, not the %d prices", pool, sum, k.Total)
		}
	}
	return o
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// refusalReason is the last fatal line of a refused run, without the scenario's file name.
func refusalReason(stderr string) string {
	var last string
	for _, line := range strings.Split(stderr, "\n") {
		if m := matrixFatal.FindStringSubmatch(line); m != nil {
			last = m[2]
		}
	}
	if last == "" {
		last = strings.TrimSpace(lastLines(stderr, 1))
	}
	last = strings.Trim(last, `"`)
	last = matrixScenarioPrefix.ReplaceAllString(last, "")
	last = strings.NewReplacer("\t", " ", "\n", " ", `\"`, `"`).Replace(last)
	if len(last) > 300 {
		last = last[:300]
	}
	return last
}

func methodsString(m map[string]int) string {
	var parts []string
	for _, name := range matrixMethods {
		if n := m[name]; n > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", name, n))
		}
	}
	return strings.Join(parts, ";")
}

// matrixCatalogModels lists the catalog's models.
func matrixCatalogModels(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(testCatalogDir(), "models"))
	if err != nil {
		t.Fatal(err)
	}
	var models []string
	for _, e := range entries {
		if e.IsDir() {
			models = append(models, e.Name())
		}
	}
	sort.Strings(models)
	return models
}

func fabricOf(chip matrixChip, roce bool) string {
	if roce {
		return "roce-" + chip.rate
	}
	return "ib-" + chip.rate
}

// search finds the cells of a model on a chip for an engine: it climbs tensor-parallel and block
// sizes over the aggregated InfiniBand cell to the first that runs or is refused for a reason
// the climb does not step past, then runs that size in every other mode and fabric.
func (r *matrixRunner) search(t *testing.T, model, engine string, chip matrixChip) []matrixCell {
	t.Helper()
	cell := func(tp, bs int, mode string, roce bool) matrixCell {
		if engine == "sglang" {
			bs = 0
		}
		return matrixCell{Model: model, Engine: engine, Chip: chip.name, Fabric: fabricOf(chip, roce), Mode: mode, TP: tp, BlockSize: bs}
	}
	blocks := matrixBlockSizes
	if engine == "sglang" {
		blocks = []int{0}
	}
	var found matrixCell
	var foundOut matrixOutcome
	var last matrixOutcome
	var lastCell matrixCell
search:
	for _, tp := range matrixTPs {
		for _, bs := range blocks {
			c := cell(tp, bs, "agg", false)
			o := r.run(t, c)
			last, lastCell = o, c
			if o.ok {
				found, foundOut = c, o
				break search
			}
			cls, _, known := classify(o.reason)
			if !known {
				t.Errorf("%s: an unclassified refusal: %s", c.key(), o.reason)
				found, foundOut = c, o
				break search
			}
			if cls.step == "block" {
				continue
			}
			if cls.step == "tp" {
				break
			}
			found, foundOut = c, o
			break search
		}
	}
	if found.Model == "" {
		found, foundOut = lastCell, last
	}
	var cells []matrixCell
	for _, mode := range matrixModes {
		for _, roce := range []bool{false, true} {
			tp := found.TP
			if mode == "pd" && !chip.statesGPN && tp < chip.gpusPerNode {
				tp = chip.gpusPerNode
			}
			c := cell(tp, found.BlockSize, mode, roce)
			o := foundOut
			if mode != "agg" || roce {
				o = r.run(t, c)
			}
			cells = append(cells, record(t, c, o))
		}
	}
	return cells
}

// record turns an outcome into a row.
func record(t *testing.T, c matrixCell, o matrixOutcome) matrixCell {
	t.Helper()
	if o.ok {
		if o.err != "" {
			t.Errorf("%s: %s", c.key(), o.err)
		}
		c.Outcome, c.Methods = "runs", methodsString(o.methods)
		return c
	}
	c.Outcome, c.Detail = "refused", o.reason
	cls, issue, known := classify(o.reason)
	if !known {
		t.Errorf("%s: an unclassified refusal: %s", c.key(), o.reason)
		c.Class = "unclassified"
		return c
	}
	c.Class = cls.id
	if cls.defect {
		c.Issue = issue
	}
	return c
}

// ---------------------------------------------------------------------------------------------
// The file

var matrixHeader = []string{"model", "engine", "chip", "fabric", "mode", "tp", "block_size", "outcome", "class", "issue", "methods", "detail"}

func (c matrixCell) fields() []string {
	return []string{c.Model, c.Engine, c.Chip, c.Fabric, c.Mode, fmt.Sprint(c.TP), fmt.Sprint(c.BlockSize), c.Outcome, c.Class, c.Issue, c.Methods, c.Detail}
}

func writeMatrixTSV(cells []matrixCell) error {
	sortCells(cells)
	var b strings.Builder
	b.WriteString(strings.Join(matrixHeader, "\t") + "\n")
	for _, c := range cells {
		b.WriteString(strings.Join(c.fields(), "\t") + "\n")
	}
	return os.WriteFile(matrixTSV, []byte(b.String()), 0o644)
}

func sortCells(cells []matrixCell) {
	sort.Slice(cells, func(i, j int) bool { return cells[i].key() < cells[j].key() })
}

func readMatrixTSV(t *testing.T) []matrixCell {
	t.Helper()
	body, err := os.ReadFile(matrixTSV)
	if err != nil {
		t.Fatalf("reading the support matrix: %v (regenerate with go test ./cmd -run ClusterMatrix -update-matrix)", err)
	}
	lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
	if strings.Join(matrixHeader, "\t") != lines[0] {
		t.Fatalf("support-matrix.tsv header is %q", lines[0])
	}
	var cells []matrixCell
	for n, line := range lines[1:] {
		f := strings.Split(line, "\t")
		if len(f) != len(matrixHeader) {
			t.Fatalf("support-matrix.tsv line %d has %d fields, want %d", n+2, len(f), len(matrixHeader))
		}
		tp, errTP := strconv.Atoi(f[5])
		bs, errBS := strconv.Atoi(f[6])
		if errTP != nil || errBS != nil {
			t.Fatalf("support-matrix.tsv line %d: tp %q and block size %q are not numbers", n+2, f[5], f[6])
		}
		cells = append(cells, matrixCell{Model: f[0], Engine: f[1], Chip: f[2], Fabric: f[3], Mode: f[4], TP: tp, BlockSize: bs,
			Outcome: f[7], Class: f[8], Issue: f[9], Methods: f[10], Detail: f[11]})
	}
	return cells
}

// ---------------------------------------------------------------------------------------------
// The published document

type tally struct{ runs, capacity, defect int }

func renderMatrixMD(cells []matrixCell) string {
	var b strings.Builder
	total := tally{}
	byEngineChip := map[string]*tally{}
	byIssue := map[string]map[string]bool{}
	issueCells := map[string]int{}
	issueClass := map[string]string{}
	capacityCells := 0
	shares := map[string]map[string]int{}
	for _, c := range cells {
		k := c.Engine + "/" + c.Chip
		if byEngineChip[k] == nil {
			byEngineChip[k] = &tally{}
		}
		t := byEngineChip[k]
		switch {
		case c.Outcome == "runs":
			total.runs++
			t.runs++
			if shares[c.Chip] == nil {
				shares[c.Chip] = map[string]int{}
			}
			for _, p := range strings.Split(c.Methods, ";") {
				if i := strings.Index(p, "="); i > 0 {
					n, err := strconv.Atoi(p[i+1:])
					if err != nil {
						panic("support-matrix.tsv methods " + c.Methods + ": " + err.Error())
					}
					shares[c.Chip][p[:i]] += n
				}
			}
		case c.Issue != "":
			total.defect++
			t.defect++
			issueCells[c.Issue]++
			issueClass[c.Issue] = c.Class
			if byIssue[c.Issue] == nil {
				byIssue[c.Issue] = map[string]bool{}
			}
			byIssue[c.Issue][c.Engine+" "+c.Model] = true
		default:
			total.capacity++
			t.capacity++
			capacityCells++
		}
	}
	fmt.Fprintf(&b, "# Support matrix\n\n")
	fmt.Fprintf(&b, "Generated by `go test ./cmd -run ClusterMatrix -update-matrix` from `docs/support-matrix.tsv`; do not edit by hand.\n\n")
	fmt.Fprintf(&b, "Each cell is one model on one chip, over one fabric, in one serving mode, on one engine, simulated with %d requests of %d prompt and %d output tokens. ", matrixRequests, matrixPromptTokens, matrixOutputTokens)
	fmt.Fprintf(&b, "A model's tensor-parallel size is the smallest of %v whose weights fit the chip, and its block size the smallest of %v the engine accepts. ", matrixTPs, matrixBlockSizes)
	fmt.Fprintf(&b, "A refusal is one of two kinds: a limit (the model does not fit the device, or vLLM's own backend for it does not support the mode; requirements section 5.2 allows both) or a defect (BLIS refuses for lack of a modelled feature; the issue that holds it is named).\n\n")
	fmt.Fprintf(&b, "## Totals\n\n| | cells |\n|---|---|\n| cells | %d |\n| simulate | %d |\n| refused, a limit (capacity or upstream) | %d |\n| refused, a defect | %d |\n\n", len(cells), total.runs, total.capacity, total.defect)
	fmt.Fprintf(&b, "## By engine and chip\n\n| engine | chip | simulate | limit | defect |\n|---|---|---|---|---|\n")
	var keys []string
	for k := range byEngineChip {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		t := byEngineChip[k]
		e := strings.SplitN(k, "/", 2)
		fmt.Fprintf(&b, "| %s | %s | %d | %d | %d |\n", e[0], e[1], t.runs, t.capacity, t.defect)
	}
	fmt.Fprintf(&b, "\n## Defects, by issue\n\n| issue | cause | cells | models |\n|---|---|---|---|\n")
	var issues []string
	for i := range byIssue {
		issues = append(issues, i)
	}
	sort.Strings(issues)
	for _, i := range issues {
		var models []string
		for m := range byIssue[i] {
			models = append(models, m)
		}
		sort.Strings(models)
		why := issueClass[i]
		for _, cl := range matrixClasses {
			if cl.id == issueClass[i] {
				why = cl.why
			}
		}
		fmt.Fprintf(&b, "| %s | %s | %d | %d |\n", i, why, issueCells[i], len(models))
	}
	fmt.Fprintf(&b, "\nThe models of each issue are the `model` and `engine` columns of `support-matrix.tsv` for its `issue`.\n")
	fmt.Fprintf(&b, "\n## Share of prices by method\n\n")
	fmt.Fprintf(&b, "The share of the prices a cell's pools hold that each method supplies, summed over the cells that simulate, by chip. It counts prices, not predicted time: the Result carries a count per method (`reports.kernel.<pool>.method_counts`) and no time per method, so a share of predicted time is not published.\n\n")
	fmt.Fprintf(&b, "| chip | %s |\n|---|%s\n", strings.Join(matrixMethods, " | "), strings.Repeat("---|", len(matrixMethods)))
	for _, chip := range matrixChips {
		s := shares[chip.name]
		sum := 0
		for _, n := range s {
			sum += n
		}
		var row []string
		for _, m := range matrixMethods {
			if sum == 0 {
				row = append(row, "-")
				continue
			}
			row = append(row, fmt.Sprintf("%.1f%%", 100*float64(s[m])/float64(sum)))
		}
		fmt.Fprintf(&b, "| %s | %s |\n", chip.name, strings.Join(row, " | "))
	}
	return b.String()
}

// ---------------------------------------------------------------------------------------------
// The tests

func TestClusterMatrix_UpdateFromSearch(t *testing.T) {
	if !*updateMatrix {
		t.Skip("rewrites docs/support-matrix.{tsv,md}; run with -update-matrix")
	}
	r := newMatrixRunner(t)
	models := matrixCatalogModels(t)
	type job struct {
		model, engine string
		chip          matrixChip
	}
	var jobs []job
	for _, e := range matrixEngines {
		for _, c := range matrixChips {
			for _, m := range models {
				jobs = append(jobs, job{m, e, c})
			}
		}
	}
	var mu sync.Mutex
	var cells []matrixCell
	next := make(chan job)
	var wg sync.WaitGroup
	for w := 0; w < matrixWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range next {
				got := r.search(t, j.model, j.engine, j.chip)
				mu.Lock()
				cells = append(cells, got...)
				mu.Unlock()
			}
		}()
	}
	for _, j := range jobs {
		next <- j
	}
	close(next)
	wg.Wait()
	if t.Failed() {
		t.Fatal("not writing the support matrix: a cell failed")
	}
	sortCells(cells)
	if err := writeMatrixTSV(cells); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(matrixMD, []byte(renderMatrixMD(cells)), 0o644); err != nil {
		t.Fatal(err)
	}
}

// M4: the recorded matrix has every cell once, and the document is what the generator renders.
func TestClusterMatrix_RecordsEveryCellOnce(t *testing.T) {
	cells := readMatrixTSV(t)
	models := matrixCatalogModels(t)
	want := len(models) * len(matrixEngines) * len(matrixChips) * 2 * len(matrixModes)
	seen := map[string]bool{}
	for _, c := range cells {
		if seen[c.key()] {
			t.Errorf("cell %s is recorded twice", c.key())
		}
		seen[c.key()] = true
	}
	if len(cells) != want {
		t.Errorf("the matrix records %d cells, want %d (%d models x %d engines x %d chips x 2 fabrics x %d modes)",
			len(cells), want, len(models), len(matrixEngines), len(matrixChips), len(matrixModes))
	}
	for _, m := range models {
		for _, e := range matrixEngines {
			for _, ch := range matrixChips {
				for _, roce := range []bool{false, true} {
					for _, mode := range matrixModes {
						k := matrixCell{Model: m, Engine: e, Chip: ch.name, Fabric: fabricOf(ch, roce), Mode: mode}.key()
						if !seen[k] {
							t.Errorf("cell %s is not recorded", k)
						}
					}
				}
			}
		}
	}
	for _, c := range cells {
		switch c.Outcome {
		case "runs":
			if c.Methods == "" || c.Class != "" || c.Issue != "" {
				t.Errorf("%s: a cell that runs states its methods and no refusal", c.key())
			}
		case "refused":
			cls, issue, ok := classify(c.Detail)
			if !ok || cls.id != c.Class {
				t.Errorf("%s: class %q does not match the recorded reason: %s", c.key(), c.Class, c.Detail)
			}
			if cls.defect && (c.Issue == "" || c.Issue != issue || strings.Contains(c.Issue, "TBD")) {
				t.Errorf("%s: a defect names its issue, got %q for %s", c.key(), c.Issue, cls.id)
			}
			if !cls.defect && c.Issue != "" {
				t.Errorf("%s: a capacity refusal carries issue %q", c.key(), c.Issue)
			}
		default:
			t.Errorf("%s: outcome %q", c.key(), c.Outcome)
		}
	}
	doc, err := os.ReadFile(matrixMD)
	if err != nil {
		t.Fatal(err)
	}
	if string(doc) != renderMatrixMD(cells) {
		t.Error("docs/support-matrix.md is not what the recorded cells render; regenerate with -update-matrix")
	}
}

// matrixSelected is the subset a pull request runs, to keep the test to a few minutes: on the
// two chips that stand for the 8-GPU boards and the NVL72 racks (h200 and gb200-nvl72), every
// InfiniBand cell and the RoCE cells of each eighth model; on the other chips, both fabrics of
// each fourth model. BLIS_MATRIX_FULL=1 runs every recorded cell.
func matrixSelected(cells []matrixCell, models []string) []matrixCell {
	if os.Getenv("BLIS_MATRIX_FULL") == "1" {
		return cells
	}
	every := func(n int) map[string]bool {
		out := map[string]bool{}
		for i, m := range models {
			if i%n == 0 {
				out[m] = true
			}
		}
		return out
	}
	eighth, fourth := every(matrixSampleEvery), every(matrixSampleEvery/2)
	var out []matrixCell
	for _, c := range cells {
		switch {
		case c.Chip == "h200" || c.Chip == "gb200-nvl72":
			if !c.isRoCE() || eighth[c.Model] {
				out = append(out, c)
			}
		case fourth[c.Model]:
			out = append(out, c)
		}
	}
	return out
}

func TestClusterMatrix_EveryCellSimulatesOrIsRefusedWithItsReason(t *testing.T) {
	if testing.Short() {
		t.Skip("simulates each cell of the support matrix")
	}
	if *updateMatrix {
		t.Skip("the update leg runs the search")
	}
	cells := readMatrixTSV(t)
	selected := matrixSelected(cells, matrixCatalogModels(t))
	r := newMatrixRunner(t)
	type result struct {
		cell matrixCell
		out  matrixOutcome
	}
	results := make(chan result, len(selected))
	next := make(chan matrixCell)
	var wg sync.WaitGroup
	for w := 0; w < matrixWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range next {
				results <- result{c, r.run(t, c)}
			}
		}()
	}
	for _, c := range selected {
		next <- c
	}
	close(next)
	wg.Wait()
	close(results)
	byKey := map[string]matrixOutcome{}
	for res := range results {
		byKey[res.cell.key()] = res.out
		c, o := res.cell, res.out
		switch c.Outcome {
		case "runs":
			if !o.ok {
				t.Errorf("%s: recorded as running, refused: %s", c.key(), o.reason)
			} else if o.err != "" {
				t.Errorf("%s: %s", c.key(), o.err)
			}
		case "refused":
			if o.ok {
				t.Errorf("%s: recorded as refused (%s), runs", c.key(), c.Class)
			} else if cls, _, ok := classify(o.reason); !ok || cls.id != c.Class {
				t.Errorf("%s: recorded as %s, refused for: %s", c.key(), c.Class, o.reason)
			}
		}
	}
	for _, c := range selected {
		if !c.isRoCE() {
			continue
		}
		twin := c
		twin.Fabric = strings.Replace(c.Fabric, "roce-", "ib-", 1)
		if ib, ok := byKey[twin.key()]; ok && ib.ok != byKey[c.key()].ok {
			t.Errorf("%s: over RoCE it runs=%v, over InfiniBand runs=%v", c.key(), byKey[c.key()].ok, ib.ok)
		}
	}
}

func TestClusterMatrix_ReadsAnEngineRefusalOfARequest(t *testing.T) {
	for _, tc := range []struct{ line, reason, issue string }{
		{`time="2026-11-20T16:12:07-05:00" level=warning msg="dropping request request_1_prefill: the engine refused it: sglang: prefill/decode disaggregation is not modelled yet (inference-sim/blis-sglang#64): request request_1_prefill"` + "\n",
			"sglang: prefill/decode disaggregation is not modelled yet (inference-sim/blis-sglang#64)", "inference-sim/blis-sglang#64"},
		{`time="2026-11-20T18:20:38-05:00" level=warning msg="dropping request request_0: the engine refused it: refused: sched: a prompt token outside the vocabulary is not modelled (inference-sim/blis-sglang#56)"` + "\n",
			"refused: sched: a prompt token outside the vocabulary is not modelled (inference-sim/blis-sglang#56)", "inference-sim/blis-sglang#56"},
	} {
		m := matrixEngineRefused.FindStringSubmatch(tc.line)
		if m == nil {
			t.Fatalf("the refusal is not read: %s", tc.line)
		}
		if m[1] != tc.reason {
			t.Errorf("read %q, want %q", m[1], tc.reason)
		}
		cls, issue, known := classify(m[1])
		if !known || !cls.defect || issue != tc.issue {
			t.Errorf("classified %q as %+v, issue %q", m[1], cls.id, issue)
		}
	}
}
