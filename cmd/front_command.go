package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/substrate/contracts/metrics"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/decision"
)

// `blis front` (design §15.5) takes the Result documents of the candidates of one decision case
// and the decision stage's artifact directory, and prints, for each point and each of the four
// fronts (XI, XT, IT, XIT), whether the point is predicted to be on the front and with what
// probability, and, for the pairs asked for, the probability that one point dominates the other.
// It reads; it writes no file and changes no Result (DEC-1).

var (
	frontArtifact string
	frontEpsilon  float64
	frontContext  string
	frontPairs    []string
	frontAllPairs bool
	frontFronts   []string
)

var frontCmd = &cobra.Command{
	Use:   "front [result-or-refusal-file ...]",
	Short: "Predict which candidates of a decision case are on each Pareto front",
	Long: "Read the Result documents of the candidates of one decision case (one model, workload and chip; " +
		"several configurations, each at several concurrencies) and the decision stage's artifact directory " +
		"(--artifact: artifact.json and model-XI.json, model-XT.json, model-IT.json, model-XIT.json). " +
		"For each point and front, print whether the point is on the predicted front (no other point dominates " +
		"it with probability above one half) and the probability that it is. --pair a:b and --all-pairs add the " +
		"probability that each point of a pair dominates the other. A point's objectives are read from its " +
		"Result: total_tokens_per_second_per_gpu, 1000 over itl_p90_ms, and ttft_p90_ms, at cluster scope. " +
		"A point the simulator did not simulate (a Result with no such figure, or a refusal document " +
		"{kind: Refusal, name, reason}) is listed with basis not_simulated and the reason, and is on no front. " +
		"--epsilon is the dominance tolerance of this command; an artifact fitted with another is refused (DEC-5). " +
		"--context names a JSON file that supplies what a Result does not hold: " +
		"{\"<result name or *>\": {\"m_tot\", \"m_act\", \"c_hbm\", \"c_bw\", \"c_fp8\", \"lisl\", \"losl\", \"src\"}}, " +
		"the case's model and chip properties as the stage's training data states them; an absent value is missing.",
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runFront(cmd.OutOrStdout(), frontOptions{
			Artifact: frontArtifact, Epsilon: frontEpsilon, Context: frontContext, Pairs: frontPairs,
			AllPairs: frontAllPairs, Fronts: frontFronts, Files: args,
		})
	},
}

func init() {
	f := frontCmd.Flags()
	f.StringVar(&frontArtifact, "artifact", "", "the decision stage's artifact directory (required)")
	f.Float64Var(&frontEpsilon, "epsilon", decision.DefaultEpsilon, "dominance tolerance; the artifact must record the same")
	f.StringVar(&frontContext, "context", "", "JSON file of case properties a Result does not hold")
	f.StringArrayVar(&frontPairs, "pair", nil, "report the probabilities for the pair a:b of point names (repeatable)")
	f.BoolVar(&frontAllPairs, "all-pairs", false, "report the probabilities of every pair of simulated points")
	f.StringSliceVar(&frontFronts, "front", nil, "limit the output to these fronts of XI, XT, IT, XIT (default all four)")
	rootCmd.AddCommand(frontCmd)
}

type frontOptions struct {
	Artifact string
	Epsilon  float64
	Context  string
	Pairs    []string
	AllPairs bool
	Fronts   []string
	Files    []string
}

// The metrics a point's objectives are read from.
const (
	frontMetricX = "total_tokens_per_second_per_gpu"
	frontMetricI = "itl_p90_ms" // interactivity is 1000 over this
	frontMetricT = "ttft_p90_ms"
)

// frontReport is the output of blis front.
type frontReport struct {
	Kind          string            `json:"kind"`
	SchemaVersion int               `json:"schema_version"`
	Artifact      frontArtifactInfo `json:"artifact"`
	Epsilon       float64           `json:"epsilon"`
	Points        []frontPoint      `json:"points"`
	Pairs         []frontPairProb   `json:"pairs,omitempty"`
}

type frontArtifactInfo struct {
	Dir              string              `json:"dir"`
	TrainingDataHash string              `json:"training_data_hash"`
	RoleCounts       decision.RoleCounts `json:"role_counts"`
	WriterCommit     string              `json:"writer_commit"`
	Features         int                 `json:"features"`
}

// Basis says where a point's values come from. Only the simulator gives values here; a point it
// did not simulate has none and is excluded with the reason (DEC-3). The calibration stage would
// give such a point estimated values, labelled "estimated"; this command does not read it.
const (
	basisSimulated    = "simulated"
	basisNotSimulated = "not_simulated"
)

type frontPoint struct {
	Name     string `json:"name"`
	Basis    string `json:"basis"`
	Excluded string `json:"excluded,omitempty"`
	// Predicted log objectives: throughput per GPU, interactivity, P90 TTFT in ms.
	LogX *float64 `json:"log_throughput_per_gpu,omitempty"`
	LogI *float64 `json:"log_interactivity,omitempty"`
	LogT *float64 `json:"log_p90_ttft_ms,omitempty"`
	// Fronts is keyed by front name.
	Fronts map[string]frontMembership `json:"fronts,omitempty"`
}

type frontMembership struct {
	// Answered is false when the simulator gave no value for an objective of the front.
	Answered    bool    `json:"answered"`
	OnFront     bool    `json:"on_front"`
	Probability float64 `json:"probability"`
}

type frontPairProb struct {
	Front       string  `json:"front"`
	A           string  `json:"a"`
	B           string  `json:"b"`
	ADominatesB float64 `json:"p_a_dominates_b"`
	BDominatesA float64 `json:"p_b_dominates_a"`
}

// refusal is the document of a point the simulator refused to simulate.
type refusal struct {
	Kind   string `yaml:"kind"`
	Name   string `yaml:"name"`
	Reason string `yaml:"reason"`
}

// caseContext holds case properties, as logs, for one point; a nil value is missing.
type caseContext struct {
	MTot *float64 `json:"m_tot"`
	MAct *float64 `json:"m_act"`
	CHBM *float64 `json:"c_hbm"`
	CBW  *float64 `json:"c_bw"`
	CFP8 *float64 `json:"c_fp8"`
	LISL *float64 `json:"lisl"`
	LOSL *float64 `json:"losl"`
	Src  *float64 `json:"src"`
}

func readContext(path string) (map[string]caseContext, error) {
	if path == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]caseContext
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("--context %s: %w", path, err)
	}
	return m, nil
}

func (c caseContext) apply(p *decision.Point) {
	set := func(dst *float64, v *float64) {
		if v != nil {
			*dst = *v
		}
	}
	set(&p.MTot, c.MTot)
	set(&p.MAct, c.MAct)
	set(&p.CHBM, c.CHBM)
	set(&p.CBW, c.CBW)
	set(&p.CFP8, c.CFP8)
	set(&p.LISL, c.LISL)
	set(&p.LOSL, c.LOSL)
	set(&p.Src, c.Src)
}

func runFront(w io.Writer, o frontOptions) error {
	if o.Artifact == "" {
		return fmt.Errorf("front: --artifact is required")
	}
	if len(o.Files) == 0 {
		return fmt.Errorf("front: give at least one Result file")
	}
	fronts := decision.Fronts
	if len(o.Fronts) > 0 {
		fronts = nil
		for _, n := range o.Fronts {
			f := decision.Front(strings.ToUpper(n))
			if !validFront(f) {
				return fmt.Errorf("front: --front %q is not one of XI, XT, IT, XIT", n)
			}
			fronts = append(fronts, f)
		}
	}
	stage, art, err := decision.LoadStage(o.Artifact, o.Epsilon)
	if err != nil {
		return fmt.Errorf("front: %w", err)
	}
	ctx, err := readContext(o.Context)
	if err != nil {
		return fmt.Errorf("front: %w", err)
	}

	var points []decision.Point
	var report []frontPoint
	var simulated []int // index into points of each simulated point, parallel to its report row
	byName := map[string]int{}
	for _, file := range o.Files {
		name, p, reason, err := readFrontPoint(file, ctx)
		if err != nil {
			return fmt.Errorf("front: %s: %w", file, err)
		}
		if _, dup := byName[name]; dup {
			return fmt.Errorf("front: two documents name the point %q", name)
		}
		byName[name] = len(report)
		row := frontPoint{Name: name, Basis: basisSimulated}
		if reason != "" {
			row.Basis, row.Excluded = basisNotSimulated, reason
		} else {
			p.ID = name
			simulated = append(simulated, len(report))
			points = append(points, p)
			row.LogX, row.LogI, row.LogT = &p.LogX, &p.LogI, &p.LogT
			row.Fronts = map[string]frontMembership{}
		}
		report = append(report, row)
	}

	pairs, err := requestedPairs(o, byName, report)
	if err != nil {
		return fmt.Errorf("front: %w", err)
	}
	out := frontReport{Kind: "FrontReport", SchemaVersion: 1, Epsilon: stage.Epsilon, Points: report,
		Artifact: frontArtifactInfo{Dir: filepath.Clean(o.Artifact), TrainingDataHash: art.TrainingDataHash,
			RoleCounts: art.RoleCounts, WriterCommit: art.WriterCommit, Features: len(art.Features)}}
	local := map[string]int{} // point name -> index in points
	for k, ri := range simulated {
		local[report[ri].Name] = k
	}
	for _, f := range fronts {
		mem, pi, err := stage.Front(f, points)
		if err != nil {
			return fmt.Errorf("front: %w", err)
		}
		for k, ri := range simulated {
			report[ri].Fronts[string(f)] = frontMembership{Answered: mem[k].Answered, OnFront: mem[k].OnFront, Probability: mem[k].Probability}
		}
		for _, pr := range pairs {
			a, b := local[pr[0]], local[pr[1]]
			if !mem[a].Answered || !mem[b].Answered {
				continue
			}
			out.Pairs = append(out.Pairs, frontPairProb{Front: string(f), A: pr[0], B: pr[1], ADominatesB: pi[a][b], BDominatesA: pi[b][a]})
		}
	}
	out.Points = report
	return writeJSON(w, out)
}

func validFront(f decision.Front) bool {
	for _, g := range decision.Fronts {
		if f == g {
			return true
		}
	}
	return false
}

// requestedPairs resolves --pair and --all-pairs to pairs of simulated point names.
func requestedPairs(o frontOptions, byName map[string]int, report []frontPoint) ([][2]string, error) {
	var out [][2]string
	for _, s := range o.Pairs {
		a, b, ok := strings.Cut(s, ":")
		if !ok || a == "" || b == "" || a == b {
			return nil, fmt.Errorf("--pair %q is not two different point names a:b", s)
		}
		for _, n := range []string{a, b} {
			i, known := byName[n]
			switch {
			case !known:
				return nil, fmt.Errorf("--pair %q: no point is named %q", s, n)
			case report[i].Basis != basisSimulated:
				return nil, fmt.Errorf("--pair %q: point %q was not simulated (%s)", s, n, report[i].Excluded)
			}
		}
		out = append(out, [2]string{a, b})
	}
	if o.AllPairs {
		var names []string
		for _, r := range report {
			if r.Basis == basisSimulated {
				names = append(names, r.Name)
			}
		}
		sort.Strings(names)
		for i := range names {
			for j := i + 1; j < len(names); j++ {
				out = append(out, [2]string{names[i], names[j]})
			}
		}
	}
	return out, nil
}

// readFrontPoint reads one candidate: a Result, or a refusal document. A non-empty reason means
// the point is excluded.
func readFrontPoint(path string, ctx map[string]caseContext) (name string, p decision.Point, reason string, err error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", p, "", err
	}
	var head refusal
	if err := yaml.Unmarshal(raw, &head); err != nil {
		return "", p, "", fmt.Errorf("not a Result or a refusal document: %w", err)
	}
	if head.Kind == "Refusal" {
		if head.Name == "" || head.Reason == "" {
			return "", p, "", fmt.Errorf("a refusal document states the point's name and the reason")
		}
		return head.Name, p, "the simulator refused it: " + head.Reason, nil
	}
	var r result.Result
	if err := result.Unmarshal(raw, &r); err != nil {
		return "", p, "", fmt.Errorf("not a Result this build reads: %w", err)
	}
	if r.Name == "" {
		return "", p, "", fmt.Errorf("the Result has no name")
	}
	p, reason = frontPointOf(&r)
	if reason == "" {
		for _, key := range []string{"*", r.Name} {
			if c, ok := ctx[key]; ok {
				c.apply(&p)
			}
		}
	}
	return r.Name, p, reason, nil
}

func logOrNaN(x float64) float64 {
	if x > 0 && !math.IsInf(x, 0) {
		return math.Log(x)
	}
	return math.NaN()
}

// frontPointOf builds the decision stage's point from a Result. The reason is non-empty, and the
// point meaningless, when the Result lacks an objective: a run that failed an acceptance condition
// of its frame reports no figures (DEV-7).
func frontPointOf(r *result.Result) (decision.Point, string) {
	figure := func(id string) (float64, string) {
		f, ok := r.Metrics[id]
		switch {
		case !ok:
			return 0, fmt.Sprintf("the Result has no %s figure", id)
		case f.Scope != metrics.ScopeCluster:
			return 0, fmt.Sprintf("%s is at scope %q; the cluster figure is needed", id, f.Scope)
		case !(f.Value > 0) || math.IsInf(f.Value, 0):
			return 0, fmt.Sprintf("%s is %v; a positive figure is needed", id, f.Value)
		}
		return f.Value, ""
	}
	x, why := figure(frontMetricX)
	if why != "" {
		return decision.Point{}, why
	}
	itl, why := figure(frontMetricI)
	if why != "" {
		return decision.Point{}, why
	}
	ttft, why := figure(frontMetricT)
	if why != "" {
		return decision.Point{}, why
	}
	p := decision.Point{LogX: math.Log(x), LogI: math.Log(1000 / itl), LogT: math.Log(ttft)}
	nan := math.NaN()
	p.MTot, p.MAct, p.CHBM, p.CBW, p.CFP8, p.LISL, p.LOSL, p.Src = nan, nan, nan, nan, nan, nan, nan, 0
	p.Engine = 0
	if r.Engine.Kind == "sglang" {
		p.Engine = 1
	}
	if c := r.Frame.Generator.Load.Concurrency; c > 0 {
		p.LConc = math.Log(float64(c))
	} else {
		p.LConc = nan
	}
	configure(&p, r)
	return p, ""
}

// configure sets the configuration features from the Result's deployment. The layout kind is
// pd when the deployment has prefill and decode pools; else dpa for data-parallel attention
// (dp above 1 with expert parallelism), ep for expert parallelism, tp otherwise. For a pd
// layout the parallel degrees are the decode pool's, as the stage's training data states them.
func configure(p *decision.Point, r *result.Result) {
	nan := math.NaN()
	gpn := r.Input.Scenario.Cluster.GPUsPerNode
	if r.Input.Packaging != nil && r.Input.Packaging.GPUsPerNode > 0 {
		gpn = r.Input.Packaging.GPUsPerNode
	}
	pools := r.Input.Deployment.Pools
	var main *deployment.Pool
	prefill, total := 0, 0
	for i := range pools {
		pl := &pools[i]
		total += pl.Nodes * gpn
		if pl.Role == deployment.RolePrefill {
			prefill += pl.Nodes * gpn
		}
		if main == nil || pl.Role == deployment.RoleDecode {
			main = pl
		}
	}
	p.TP, p.EP, p.DP, p.GPUs = nan, nan, nan, nan
	p.Seqs, p.Btok = nan, nan
	if main == nil {
		return
	}
	par := main.Parallel
	ep := 1
	if par.EnableExpertParallel {
		ep = par.TP * max(par.DP, 1)
	}
	p.TP, p.EP, p.DP = logOrNaN(float64(max(par.TP, 1))), logOrNaN(float64(ep)), logOrNaN(float64(max(par.DP, 1)))
	if total > 0 {
		p.GPUs = math.Log(float64(total))
	}
	switch {
	case hasRoles(pools, deployment.RolePrefill, deployment.RoleDecode):
		p.Kind, p.PD = decision.KindPD, 1
		if total > 0 {
			p.PDFrac = float64(prefill) / float64(total)
		}
	case par.EnableExpertParallel && par.DP > 1:
		p.Kind = decision.KindDPA
	case par.EnableExpertParallel:
		p.Kind = decision.KindEP
	default:
		p.Kind = decision.KindTP
	}
	if n := main.Engine.Node(); n != nil {
		if v, ok := engineInt(n, "max_num_seqs"); ok {
			p.Seqs = math.Log(float64(max(v, 1)))
		}
		if v, ok := engineInt(n, "max_num_batched_tokens"); ok {
			p.Btok = math.Log(float64(max(v, 1)))
		}
		if hasKey(n, "speculative_config") {
			p.Spec = 1
		}
	}
	for _, wp := range r.Input.WorkloadProperties {
		if wp.Name == result.SpeculativeAcceptance {
			p.Spec = 1
		}
	}
}

func hasRoles(pools []deployment.Pool, roles ...deployment.Role) bool {
	for _, want := range roles {
		found := false
		for _, pl := range pools {
			found = found || pl.Role == want
		}
		if !found {
			return false
		}
	}
	return true
}

func hasKey(n *yaml.Node, key string) bool {
	for i := 0; n.Kind == yaml.MappingNode && i+1 < len(n.Content); i += 2 {
		if n.Content[i].Value == key && n.Content[i+1].Tag != "!!null" {
			return true
		}
	}
	return false
}

func engineInt(n *yaml.Node, key string) (int, bool) {
	for i := 0; n.Kind == yaml.MappingNode && i+1 < len(n.Content); i += 2 {
		if n.Content[i].Value == key {
			var v int
			if n.Content[i+1].Decode(&v) == nil {
				return v, true
			}
		}
	}
	return 0, false
}
